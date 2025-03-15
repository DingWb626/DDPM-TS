import argparse

parser = argparse.ArgumentParser()


# basic config
parser.add_argument(
    "--task_name",
    type=str,
    required=True,
    default="pretrain",
    help="task name, options:[pretrain, finetune]",
)
parser.add_argument("--downstream_task", type=str, default="forecast", help="downstream task, options:[forecasting, classification]")
parser.add_argument("--is_training", type=int, default=1, help="status")


# data loader
parser.add_argument(
    "--data", type=str, required=True, default="ETTh1", help="dataset type"
)
parser.add_argument(
    "--root_path", type=str, default="./datasets", help="root path of the data file"
)
parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
)
parser.add_argument(
    "--target", type=str, default="OT", help="target feature in S or MS task"
)
parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
)

parser.add_argument(
    "--select_channels",
    type=float,
    default=1,
    help="select the rate of channels to train",
)
parser.add_argument(
    "--use_norm",
    type=int,
    default=1,
    help="use normalization",
)
parser.add_argument(
    "--embed",
    type=str,
    default="timeF",
    help="time features encoding, options:[timeF, fixed, learned]",
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size of train input data"
)

# Classification
parser.add_argument("--num_classes", type=int, default=6, help="number of classes")

# forecasting task
parser.add_argument("--seq_len", type=int, default=336, help="input sequence length")
parser.add_argument("--input_len", type=int, default=336, help="input sequence length")
parser.add_argument("--label_len", type=int, default=0, help="start token length")
parser.add_argument(
    "--pred_len", type=int, default=96, help="prediction sequence length"
)
parser.add_argument(
    "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
)

# optimization
parser.add_argument(
    "--num_workers", type=int, default=5, help="data loader num workers"
)


