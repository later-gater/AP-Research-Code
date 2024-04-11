import sys
import access_permutations
import market

def main(start_row: int, end_row: int):
    rows = access_permutations.access_rows(slice(start_row, end_row))
    for index, row in rows.iterrows():
        df = market.run_market(index, row["file_name"], row["alpha_2"], row["eos_1"], row["eos_2"], row["robot_growth_func"], row["rate_k"], row["rate_z"], row["incomes"])
        df.to_pickle(f"data/{index}_{row['file_name']}.pkl")
        print(f"Market {row['file_name']} finished")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))