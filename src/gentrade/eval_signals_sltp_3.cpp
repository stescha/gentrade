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

py::array_t<double> cpp2py_double(std::vector<double> vec, int arr_size) {
  auto result = py::array_t<double>(arr_size);
  auto result_buffer = result.request();
  double *result_ptr = (double *)result_buffer.ptr;

  std::memcpy(result_ptr, vec.data(), vec.size() * sizeof(double));
  return result;
}

py::array_t<int> cpp2py_int(std::vector<int> vec, int arr_size) {
  auto result = py::array_t<int>(arr_size);
  auto result_buffer = result.request();
  int *result_ptr = (int *)result_buffer.ptr;

  std::memcpy(result_ptr, vec.data(), vec.size() * sizeof(int));
  return result;
}

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>, py::array_t<int>, py::array_t<double>>
eval_sltp(py::array_t<double, py::array::c_style | py::array::forcecast> open_arr,
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
  double tp_price = 0.0;
  double sl_price = 0.0;
  double highest_high = 0.0;

  for (int t = 0; t < n - 1; ++t) {
    values[t] = balance + position * close[t];
    positions[t] = position;

    if (position == 0 && buys[t] == 1) {
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
        highest_high = high[t];
      }
      continue;
    }

    if (position > 0) {
      bool sell_signal = sells[t] == 1;

      if (sl_enabled && sl_trail_enabled) {
        highest_high = std::max(highest_high, high[t]);
        sl_price = highest_high * (1.0 - sl_stop_value);
      }

      bool tp_hit = tp_enabled && high[t] >= tp_price;
      bool sl_hit = sl_enabled && low[t] <= sl_price;
      bool should_exit = sell_signal || tp_hit || sl_hit;

      if (!should_exit || t + order_delay >= n) {
        continue;
      }

      sell_price = open[t + order_delay];
      balance += sell_price * position * (1.0 - sell_fee);
      pnl = ((sell_price - buy_price) - (sell_fee * sell_price + buy_fee * buy_price)) / buy_price;
      position = 0;
      sell_times.push_back(t + order_delay);
      buy_times.push_back(buy_time);
      trade_returns.push_back(pnl);
    }
  }

  int t = n - 1;
  values[t] = balance + position * close[t];
  positions[t] = position;

  if (position > 0) {
    sell_price = open[t];
    balance += sell_price * position * (1.0 - sell_fee);
    pnl = ((sell_price - buy_price) - (sell_fee * sell_price + buy_fee * buy_price)) / buy_price;
    trade_returns.push_back(pnl);
    position = 0;
    buy_times.push_back(buy_time);
    sell_times.push_back(t);
    values[t] = balance;
  } else if (n > 1) {
    values[n - 1] = values[n - 2];
  }

  return std::make_tuple(cpp2py_int(buy_times, buy_times.size()),
                         cpp2py_int(sell_times, sell_times.size()),
                         cpp2py_double(values, values.size()),
                         cpp2py_int(positions, positions.size()),
                         cpp2py_double(trade_returns, trade_returns.size()));
}

PYBIND11_MODULE(eval_signals_sltp_3, m) {
  m.doc() = "SL/TP backtester plugin";
  m.def("eval_sltp",
        &eval_sltp,
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
