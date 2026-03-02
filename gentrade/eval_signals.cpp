#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

#include <stdio.h>


// g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` eval_signals.cpp -o eval_signals`python3-config --extension-suffix`


namespace py = pybind11;

py::array_t<double> cpp2py_double(std::vector<double> vec, int arr_size){
  auto result        = py::array_t<double>(arr_size);
  auto result_buffer = result.request();
  double *result_ptr    = (double *) result_buffer.ptr;

  // copy std::vector -> py::array
  std::memcpy(result_ptr, vec.data(),vec.size()*sizeof(double));
  return result;
}


py::array_t<int> cpp2py_int(std::vector<int> vec, int arr_size){
  auto result        = py::array_t<int>(arr_size);
  auto result_buffer = result.request();
  int *result_ptr    = (int *) result_buffer.ptr;

  // copy std::vector -> py::array
  std::memcpy(result_ptr, vec.data(),vec.size()*sizeof(int));
  return result;
}


std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>, py::array_t<int>, py::array_t<double>>
          eval(py::array_t<double, py::array::c_style | py::array::forcecast> open_arr,
                         py::array_t<double, py::array::c_style | py::array::forcecast> close_arr,
                         py::array_t<int, py::array::c_style | py::array::forcecast> buys_arr,
                         py::array_t<int, py::array::c_style | py::array::forcecast> sells_arr,
                         double buy_fee,
                         double sell_fee
                        )
{
  std::vector<double> open(open_arr.size());
  std::vector<double> close(close_arr.size());
  std::vector<int> buys(buys_arr.size());
  std::vector<int> sells(sells_arr.size());

  // copy py::array -> std::vector
  std::memcpy(open.data(),open_arr.data(),open_arr.size()*sizeof(double));
  std::memcpy(close.data(),close_arr.data(),close_arr.size()*sizeof(double));
  std::memcpy(buys.data(),buys_arr.data(),buys_arr.size()*sizeof(int));
  std::memcpy(sells.data(),sells_arr.data(),sells_arr.size()*sizeof(int));

  // call pure C++ function
  std::vector<int> buy_times(0);
  std::vector<int> sell_times(0);
  std::vector<double> pnls(0);
  std::vector<double> values(close_arr.size());
  std::vector<int> positions(close_arr.size(), 0);

  int order_delay = 1;
  int position = 0;
  double balance = 0;
  double buy_price;
  double sell_price;
  double asset = 0;
  double pnl = 0;
  int buy_time = 0;
  bool in_trade = false;

  for(int t=0; t< close_arr.size()-2; ++t){
      values[t] = balance + position*close[t];
      positions[t] = position;
//      if (in_trade){
//        // printf("%f\n", fiat);
//        fiats[t] = fiat + position*close[t];
//      } else {
//        fiats[t] = fiat;
//      }
      if ((buys[t] == 1) && position==0) {
        buy_price = open[t + order_delay];
        position = 1;
        balance -= buy_price * (1 + buy_fee);
        buy_time = t + order_delay;
      } else if((sells[t] == 1) && position>0 ){
        sell_price = open[t+order_delay];
        balance += sell_price * position * (1 - sell_fee);
        pnl = ((sell_price - buy_price) - (sell_fee*sell_price + buy_fee*buy_price))/buy_price;
        position = 0;
        sell_times.push_back(t+order_delay);
        buy_times.push_back(buy_time);
        pnls.push_back(pnl);

      }

  }

  int t = close.size()-2;
  values[t] = balance + position*close[t];
  positions[t] = position;
  if(position>0){
    t = close.size() - 1;
    sell_price = open[t];
    balance += sell_price*position*(1-sell_fee);
    pnl = ((sell_price - buy_price) - (sell_fee*sell_price + buy_fee*buy_price))/buy_price;
    pnls.push_back(pnl);
    position = 0;
    buy_times.push_back(buy_time);
    sell_times.push_back(t);
    values[t] = balance;
  } else {
    values[close.size()-1] = values[close.size()-2];
  }
  return std::make_tuple(cpp2py_int(buy_times, buy_times.size()),
                         cpp2py_int(sell_times, sell_times.size()),
                         cpp2py_double(values, values.size()),
                         cpp2py_int(positions, positions.size()),
                         cpp2py_double(pnls, pnls.size()));
}


std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>, py::array_t<int>, py::array_t<double>>
          eval_intrade(py::array_t<double, py::array::c_style | py::array::forcecast> open_arr,
                         py::array_t<double, py::array::c_style | py::array::forcecast> close_arr,
                         py::array_t<int, py::array::c_style | py::array::forcecast> buys_arr,
                         py::array_t<int, py::array::c_style | py::array::forcecast> sells_arr,
                         double buy_fee,
                         double sell_fee,
                         const std::function<bool(double, double, double)> &in_trade_func
                        )
{
  std::vector<double> open(open_arr.size());
  std::vector<double> close(close_arr.size());
  std::vector<int> buys(buys_arr.size());
  std::vector<int> sells(sells_arr.size());

  // copy py::array -> std::vector
  std::memcpy(open.data(),open_arr.data(),open_arr.size()*sizeof(double));
  std::memcpy(close.data(),close_arr.data(),close_arr.size()*sizeof(double));
  std::memcpy(buys.data(),buys_arr.data(),buys_arr.size()*sizeof(int));
  std::memcpy(sells.data(),sells_arr.data(),sells_arr.size()*sizeof(int));

  // call pure C++ function
  std::vector<int> buy_times(0);
  std::vector<int> sell_times(0);
  std::vector<double> pnls(0);
  std::vector<double> values(close_arr.size());
  std::vector<int> positions(close_arr.size(), 0);

  int order_delay = 1;
  int position = 0;
  double balance = 0;
  double buy_price;
  double sell_price;
  double asset = 0;
  double pnl = 0;
  int buy_time = 0;
  bool in_trade = false;

  for(int t=0; t< close_arr.size()-2; ++t){
      values[t] = balance + position*close[t];
      positions[t] = position;
//      if (in_trade){
//        // printf("%f\n", fiat);
//        fiats[t] = fiat + position*close[t];
//      } else {
//        fiats[t] = fiat;
//      }
      if ((buys[t] == 1) && position==0) {
        buy_price = open[t + order_delay];
        position = 1;
        balance -= buy_price * (1 + buy_fee);
        buy_time = t + order_delay;
      } else if((sells[t] == 1) && position>0 ){
        sell_price = open[t+order_delay];
        balance += sell_price * position * (1 - sell_fee);
        pnl = ((sell_price - buy_price) - (sell_fee*sell_price + buy_fee*buy_price))/buy_price;
        position = 0;
        sell_times.push_back(t+order_delay);
        buy_times.push_back(buy_time);
        pnls.push_back(pnl);

      }

  }

  int t = close.size()-2;
  values[t] = balance + position*close[t];
  positions[t] = position;
  if(position>0){
    t = close.size() - 1;
    sell_price = open[t];
    balance += sell_price*position*(1-sell_fee);
    pnl = ((sell_price - buy_price) - (sell_fee*sell_price + buy_fee*buy_price))/buy_price;
    pnls.push_back(pnl);
    position = 0;
    buy_times.push_back(buy_time);
    sell_times.push_back(t);
    values[t] = balance;
  } else {
    values[close.size()-1] = values[close.size()-2];
  }
  return std::make_tuple(cpp2py_int(buy_times, buy_times.size()),
                         cpp2py_int(sell_times, sell_times.size()),
                         cpp2py_double(values, values.size()),
                         cpp2py_int(positions, positions.size()),
                         cpp2py_double(pnls, pnls.size()));
}



// wrap as Python module
PYBIND11_MODULE(eval_signals,m)
{
  m.doc() = "pybind11 example plugin";
  m.def("eval", &eval, "");
  m.def("eval_intrade", &eval_intrade, "");

}
