// C++/pybind11 implementation of the SL/TP backtester.
// The logic mirrors vectorbt's `from_signals` semantics so tests can compare
// time-aligned fills, trailing stops, and explicit exit signals without
// falling back to Python loops.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// Helper for copying std::vector<double> into a new NumPy array that keeps
// ownership on the Python side.
py::array_t<double> cpp2py_double(std::vector<double> vec, int arr_size) {
  auto result = py::array_t<double>(arr_size);
  auto result_buffer = result.request();
  double *result_ptr = (double *)result_buffer.ptr;

  std::memcpy(result_ptr, vec.data(), vec.size() * sizeof(double));
  return result;
}

// Same helper for integer payloads (indices, positions, etc.).
py::array_t<int> cpp2py_int(std::vector<int> vec, int arr_size) {
  auto result = py::array_t<int>(arr_size);
  auto result_buffer = result.request();
  int *result_ptr = (int *)result_buffer.ptr;

  std::memcpy(result_ptr, vec.data(), vec.size() * sizeof(int));
  return result;
}

/**
 * Core SL/TP evaluator.
 *
 * The implementation emulates vectorbt's behaviour with a fixed
 * 1-bar order delay, inclusive intra-bar stop detection, optional
 * trailing stops, and a small state machine that gates manual exit
 * signals until the next bar's open. The return values intentionally
 * expose intermediate arrays to simplify parity assertions in Python.
 */
std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>, py::array_t<int>, py::array_t<double>>
eval_sltp_2(py::array_t<double, py::array::c_style | py::array::forcecast> open_arr,
          py::array_t<double, py::array::c_style | py::array::forcecast> high_arr,
          py::array_t<double, py::array::c_style | py::array::forcecast> low_arr,
          py::array_t<double, py::array::c_style | py::array::forcecast> close_arr,
          py::array_t<int, py::array::c_style | py::array::forcecast> buys_arr,
          py::object sells_arr,
          double buy_fee,
          double sell_fee,
          py::object tp_stop,
          py::object sl_stop,
          py::object sl_trail) {
  if (open_arr.size() != high_arr.size() || open_arr.size() != low_arr.size() ||
      open_arr.size() != close_arr.size() || open_arr.size() != buys_arr.size()) {
    throw std::runtime_error("All OHLC and signal arrays must have equal length.");
  }

  if (open_arr.size() < 2) {
    throw std::runtime_error("Input arrays must contain at least 2 rows.");
  }

  const int n = static_cast<int>(open_arr.size());

  std::vector<double> open(open_arr.size());
  std::vector<double> high(high_arr.size());
  std::vector<double> low(low_arr.size());
  std::vector<double> close(close_arr.size());
  std::vector<int> buys(buys_arr.size());
  std::vector<int> sells(open_arr.size(), 0);

  std::memcpy(open.data(), open_arr.data(), open_arr.size() * sizeof(double));
  std::memcpy(high.data(), high_arr.data(), high_arr.size() * sizeof(double));
  std::memcpy(low.data(), low_arr.data(), low_arr.size() * sizeof(double));
  std::memcpy(close.data(), close_arr.data(), close_arr.size() * sizeof(double));
  std::memcpy(buys.data(), buys_arr.data(), buys_arr.size() * sizeof(int));

  if (!sells_arr.is_none()) {
    py::array_t<int, py::array::c_style | py::array::forcecast> sells_arr_typed =
        py::cast<py::array_t<int, py::array::c_style | py::array::forcecast>>(sells_arr);
    if (sells_arr_typed.size() != open_arr.size()) {
      throw std::runtime_error("sells_arr must be None or have same length as OHLC arrays.");
    }
    std::memcpy(sells.data(), sells_arr_typed.data(), sells_arr_typed.size() * sizeof(int));
  }

  bool tp_enabled = !tp_stop.is_none();
  bool sl_enabled = !sl_stop.is_none();

  double tp_stop_value = 0.0;
  double sl_stop_value = 0.0;

  if (tp_enabled) {
    tp_stop_value = py::cast<double>(tp_stop);
    if (tp_stop_value <= 0.0) {
      throw std::runtime_error("tp_stop must be > 0 when provided.");
    }
  }

  if (sl_enabled) {
    sl_stop_value = py::cast<double>(sl_stop);
    if (sl_stop_value <= 0.0) {
      throw std::runtime_error("sl_stop must be > 0 when provided.");
    }
  }

  bool sl_trail_enabled = false;
  if (!sl_trail.is_none() && sl_enabled) {
    sl_trail_enabled = py::cast<bool>(sl_trail);
  }

  std::vector<int> buy_times(0);
  std::vector<int> sell_times(0);
  std::vector<double> trade_returns(0);
  std::vector<double> values(close_arr.size());
  std::vector<int> positions(close_arr.size(), 0);

  int order_delay = 1;
  int position = 0;
  double balance = 0.0;
  double buy_price = 0.0;
  double sell_price = 0.0;
  double pnl = 0.0;
  int buy_time = 0;
  int stop_activation_time = 0;
  bool exit_signal_pending = false;
  int exit_execution_time = -1;
  double tp_price = 0.0;
  double sl_price = 0.0;
  double highest_high = 0.0;

  for (int t = 0; t < n - 1; ++t) {
    values[t] = balance + position * close[t];
    positions[t] = position;

    const bool entry_signal = buys[t] == 1;

    if (position > 0) {
      // Queue manual exits so they fill on the next bar unless a stop fires sooner.
      if (sells[t] == 1 && !exit_signal_pending) {
        int execution_time = t + order_delay;
        if (execution_time < n) {
          exit_signal_pending = true;
          exit_execution_time = execution_time;
        }
      }

      // Stops only become eligible after the entry candle fully closes.
      if (t < stop_activation_time) {
        continue;
      }

      double trailing_high = highest_high;
      if (sl_enabled && sl_trail_enabled) {
        trailing_high = std::max(highest_high, high[t]);
      }

      bool tp_hit = tp_enabled && high[t] >= tp_price;
      bool sl_hit = sl_enabled && low[t] <= sl_price;
      bool signal_ready = exit_signal_pending && t >= exit_execution_time;
      int execution_index = -1;
      double execution_price = 0.0;
      bool processed_exit = false;

      // Stops fire intra-bar: if open gaps beyond the target, use the stop level,
      // otherwise use the first price (open) that crosses the threshold.
      if (sl_hit) {
        execution_index = t;
        execution_price = open[t] <= sl_price ? open[t] : sl_price;
        processed_exit = true;
        exit_signal_pending = false;
        exit_execution_time = -1;
      } else if (tp_hit) {
        execution_index = t;
        execution_price = open[t] >= tp_price ? open[t] : tp_price;
        processed_exit = true;
        exit_signal_pending = false;
        exit_execution_time = -1;
      } else if (signal_ready) {
        execution_index = t;
        execution_price = open[t];
        processed_exit = true;
        exit_signal_pending = false;
        exit_execution_time = -1;
      }

      // No exit yet: update trailing stop anchor and continue scanning.
      if (!processed_exit) {
        if (sl_enabled && sl_trail_enabled) {
          highest_high = trailing_high;
          sl_price = highest_high * (1.0 - sl_stop_value);
        }
        continue;
      }

      // Exit accounting mirrors vectorbt: proceeds posted to cash immediately.
      balance += execution_price * position * (1.0 - sell_fee);
      pnl = ((execution_price - buy_price) -
             (sell_fee * execution_price + buy_fee * buy_price)) /
            buy_price;
      position = 0;
      sell_times.push_back(execution_index);
      buy_times.push_back(buy_time);
      trade_returns.push_back(pnl);

      // Always reflect the closed position in the equity curve for this bar.
      values[t] = balance;
      positions[t] = position;
      stop_activation_time = 0;
    }

    // Entry path: execute buys on the next bar's open to avoid look-ahead.
    if (position == 0 && entry_signal) {
      if (t + order_delay >= n) {
        continue;
      }

      buy_price = open[t + order_delay];
      position = 1;
      balance -= buy_price * (1.0 + buy_fee);
      buy_time = t + order_delay;

      if (tp_enabled) {
        tp_price = buy_price * (1.0 + tp_stop_value);
      }

      if (sl_enabled) {
        sl_price = buy_price * (1.0 - sl_stop_value);
        highest_high = buy_price;
      }
      stop_activation_time = buy_time + 1;
      exit_signal_pending = false;
      exit_execution_time = -1;
      continue;
    }
  }

  int t = n - 1;
  values[t] = balance + position * close[t];
  positions[t] = position;

  // Force liquidation at the final close so the Python harness observes
  // the same terminal equity as vectorbt.
  if (position > 0) {
    sell_price = close[t];
    balance += sell_price * position * (1.0 - sell_fee);
    pnl = ((sell_price - buy_price) - (sell_fee * sell_price + buy_fee * buy_price)) / buy_price;
    trade_returns.push_back(pnl);
    position = 0;
    buy_times.push_back(buy_time);
    sell_times.push_back(t);
    values[t] = balance;
  }

  return std::make_tuple(cpp2py_int(buy_times, buy_times.size()),
                         cpp2py_int(sell_times, sell_times.size()),
                         cpp2py_double(values, values.size()),
                         cpp2py_int(positions, positions.size()),
                         cpp2py_double(trade_returns, trade_returns.size()));
}

PYBIND11_MODULE(eval_signals_sltp_2, m) {
  m.doc() = "SL/TP backtester plugin";
  m.def("eval_sltp_2",
        &eval_sltp_2,
        py::arg("open_arr"),
        py::arg("high_arr"),
        py::arg("low_arr"),
        py::arg("close_arr"),
        py::arg("buys_arr"),
        py::arg("sells_arr") = py::none(),
        py::arg("buy_fee") = 0.0,
        py::arg("sell_fee") = 0.0,
        py::arg("tp_stop") = py::none(),
        py::arg("sl_stop") = py::none(),
        py::arg("sl_trail") = py::none(),
        "");
}
