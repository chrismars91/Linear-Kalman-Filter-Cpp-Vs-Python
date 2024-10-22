#include <pybind11/pybind11.h>
#include "kf.h"
#include <pybind11/eigen.h>
#include <Eigen/Dense>

long double getPI(){return 3.141592653589793238462643383279;}


PYBIND11_MODULE(kalmanfilter_cpp, m) {
		m.doc() = "pybind11 example plugin"; // optional module docstring
		m.def("getPI", &getPI, "getPI");
		pybind11::class_<KF>(m, "KF")
			.def(pybind11::init<double, double>())
			.def("predict", &KF::predict)
			.def("update", &KF::update)
			.def("create_matrix", &KF::create_matrix, "Create a random Eigen matrix")
			.def("get_prediction", &KF::get_prediction, "gets prediction")
			.def("get_K", &KF::get_K, "gets K")
			.def("get_P", &KF::get_P, "gets P");
		m.attr("__version__") = "0.0.1";
}