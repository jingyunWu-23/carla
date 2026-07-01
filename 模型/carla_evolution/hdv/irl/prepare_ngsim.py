#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "模型/dataset/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv"
DEFAULT_OUTPUT = "模型/carla_evolution/results/hdv_irl/ngsim_expert.csv"


def numeric_series(series: pd.Series, scale: float = 1.0) -> pd.Series:
    values = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(values, errors="coerce") * scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw NGSIM CSV for CARLA HDV IRL.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Raw NGSIM CSV path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Converted expert CSV path.")
    parser.add_argument("--location", default="us-101", choices=["us-101", "i-80", "all"])
    parser.add_argument("--max-vehicles", type=int, default=500)
    parser.add_argument("--min-frames", type=int, default=80)
    parser.add_argument("--chunksize", type=int, default=300000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw NGSIM CSV not found: {input_path}. "
            "Replace the placeholder path with the real downloaded NGSIM CSV path."
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

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
    first_write = True
    written = 0

    with input_path.open("r", encoding="utf-8") as input_file:
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
                    "time": numeric_series(chunk["Global_Time"], scale=1.0 / 1000.0),
                    "x": numeric_series(chunk["Local_Y"], scale=0.3048),
                    "y": numeric_series(chunk["Local_X"], scale=0.3048),
                    "speed": numeric_series(chunk["v_Vel"], scale=0.3048),
                    "acc": numeric_series(chunk["v_Acc"], scale=0.3048),
                    "lane_id": numeric_series(chunk["Lane_ID"]),
                    "front_gap": numeric_series(chunk["Space_Headway"], scale=0.3048),
                    "thw_front": numeric_series(chunk["Time_Headway"]),
                }
            )
            out = out.dropna(subset=["time", "x", "y", "speed"])
            out = out.sort_values(["episode", "agent_id", "time"])
            out.to_csv(output, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
            written += len(out)

    print(f"Saved converted NGSIM trajectories to: {output}")
    print(f"Rows written: {written}")
    print(f"Vehicles selected: {len(selected_vehicle_ids or [])}")


if __name__ == "__main__":
    main()
