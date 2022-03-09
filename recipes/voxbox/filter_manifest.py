import os.path
from argparse import ArgumentParser

import numpy as np


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--snr_file", type=str, required=True)
    parser.add_argument("--manifests_fld", type=str, required=True)
    parser.add_argument("--snr_threshold", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = get_args()
    snr_data = np.load(args.snr_file)

    snr_data = [(os.path.splitext(os.path.basename(pth))[0], snr) for pth, snr in snr_data]
    filtered_snr = {pth for pth, snr in snr_data if float(snr) < args.snr_threshold}
    with open(os.path.join(args.manifests_fld, "metadata_train.txt")) as d_in:
        train_lines = [(line.split("||")[0], line.replace("\n", "")) for line in d_in]

    with open(os.path.join(args.manifests_fld, "metadata_test.txt")) as d_in:
        test_lines = [(line.split("||")[0], line.replace("\n", "")) for line in d_in]

    filtered_train = [it for it in train_lines if it[0] not in filtered_snr]
    filtered_test = [it for it in test_lines if it[0] not in filtered_snr]

    with open(os.path.join(args.manifests_fld, "metadata_train_filtered.txt"), "w") as d_out:
        for pth, line in filtered_train:
            print(line, file=d_out)

    with open(os.path.join(args.manifests_fld, "metadata_test_filtered.txt"), "w") as d_out:
        for pth, line in filtered_test:
            print(line, file=d_out)


if __name__ == '__main__':
    main()
