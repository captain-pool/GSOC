py_library(
  name = "main"
  srcs = ["main.py"]
  deps = [
     "//external:tensorflow_hub",
     ":expect_tensorflow_installed",
     ":expect_tensorflow_datasets_installed"
  ],
)

sh_binary(
  name = "train",
  srcs = ["train.sh"],
  deps = [
      ":main"
  ],
)

py_library(
    name = "expect_tensorflow_installed"
)

py_library(
    name = "expect_tensorflow_datasets_installed"
)
