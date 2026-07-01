"""
Convert the large NGSIM trajectory CSV into a smaller CSV for IRL reward learning.

The output schema is compatible with learn_hdv_aggressive_reward.py:
episode, agent_id, time, x, y, speed, front_gap, thw_front

Example:
    python prepare_ngsim_for_irl.py --location us-101 --max-vehicles 300
"""

import argparse
import os

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


DEFAULT_INPUT = os.path.join(
    PROJECT_DIR,
    "dataset",
    "Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare NGSIM CSV for HDV IRL.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Raw NGSIM CSV path.")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(SCRIPT_DIR, "hdv_reward_irl", "ngsim_expert.csv"),
        help="Converted CSV path.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="us-101",
        choices=["us-101", "i-80", "all"],
        help="NGSIM location to keep.",
    )
    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=300,
        help="Maximum number of vehicles to keep after filtering.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=80,
        help="Minimum trajectory length per vehicle.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=300000,
        help="Rows per pandas chunk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    columns = [
        "Vehicle_ID",
        "Frame_ID",
        "Global_Time",
        "Local_X",
        "Local_Y",
        "v_Vel",
        "v_Acc",
        "Lane_ID",
        "Space_Headway",
        "Time_Headway",
        "Location",
    ]

    selected_vehicle_ids = None
    written = 0
    first_write = True

    with open(args.input, "r", encoding="utf-8") as input_file:
        reader = pd.read_csv(input_file, usecols=columns, chunksize=args.chunksize)
        for chunk in reader:
            if args.location != "all":
                chunk = chunk[chunk["Location"].astype(str).str.lower() == args.location]
            if chunk.empty:
                continue

            chunk = chunk.drop_duplicates(subset=["Vehicle_ID", "Frame_ID", "Location"])
            counts = chunk.groupby("Vehicle_ID")["Frame_ID"].count()
            valid_ids = set(counts[counts >= args.min_frames].index.tolist())

            if selected_vehicle_ids is None:
                selected_vehicle_ids = set(list(valid_ids)[: args.max_vehicles])
            else:
                remaining = args.max_vehicles - len(selected_vehicle_ids)
                if remaining > 0:
                    selected_vehicle_ids.update(list(valid_ids - selected_vehicle_ids)[:remaining])

            chunk = chunk[chunk["Vehicle_ID"].isin(selected_vehicle_ids)]
            if chunk.empty:
                continue

            out = pd.DataFrame(
                {
                    "episode": chunk["Location"].astype(str),
                    "agent_id": chunk["Vehicle_ID"].astype(str),
                    "time": chunk["Global_Time"].astype(float) / 1000.0,
                "x": chunk["Local_Y"].astype(float) * 0.3048,
                "y": chunk["Local_X"].astype(float) * 0.3048,
                    "speed": chunk["v_Vel"].astype(float) * 0.3048,
                    "acc": chunk["v_Acc"].astype(float) * 0.3048,
                    "lane_id": chunk["Lane_ID"],
                    "front_gap": pd.to_numeric(chunk["Space_Headway"], errors="coerce") * 0.3048,
                    "thw_front": pd.to_numeric(chunk["Time_Headway"], errors="coerce"),
                }
            )
            out = out.sort_values(["episode", "agent_id", "time"])
            out.to_csv(args.output, mode="w" if first_write else "a",
                       header=first_write, index=False, encoding="utf-8")
            first_write = False
            written += len(out)

    print("Saved converted NGSIM trajectories to: {}".format(args.output))
    print("Rows written: {}".format(written))
    print("Vehicles selected: {}".format(len(selected_vehicle_ids or [])))


if __name__ == "__main__":
    main()
