from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd


def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in-path', type=str, required=True, help='Tensorboard event files or a single tensorboard '
                                                                   'file location')
    parser.add_argument('--ex-path', type=str, required=True, help='location to save the exported data')

    args = parser.parse_args()
    event_data = event_accumulator.EventAccumulator(args.in_path)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    key = "ray/tune/episode_reward_mean"
    key = "comm_number"
    df = pd.DataFrame(columns=[key])
    df[key] = pd.DataFrame(event_data.Scalars(key)).value

    df.to_csv(args.ex_path)

    print("Tensorboard data exported successfully")


if __name__ == '__main__':
    main()
