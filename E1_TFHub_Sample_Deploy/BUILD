py_binary(
    name = "export",
    srcs = ["export.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":export_lib"]
)
py_library(
    name = "export_lib",
    srcs = ["export.py"],
    srcs_version = "PY2AND3",
)
py_test(
    name = "export_test",
    srcs = ["export_test.py"],
    data = glob(["test_data/*.jpg"]),
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":export_lib", ":numpy_installed", ":tensorflow_installed"]
)
py_library(
    name = "numpy_installed"
)
py_library(
    name = "tensorflow_installed"
)
