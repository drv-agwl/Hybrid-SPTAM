import json

SEMANTIC_DATABASE_FILE = "world_frame_set_1.txt"
SEMANTIC_DATABASE_AVERAGE_FILE = "world_frame_set_1_average.txt"
database_entries = {}
with open(SEMANTIC_DATABASE_FILE, "r") as f:
    for line in f:
        database_entry = json.loads(line)
        if database_entry["Name"] in database_entries:
            if database_entry["Corner_name"] not in database_entries[database_entry["Name"]]:
                database_entries[database_entry["Name"]][database_entry["Corner_name"]] = {}
                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_X"] = database_entry["X"] * database_entry["confidence"]
                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_X"] = 1 + database_entry["confidence"]
                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_Y"] = database_entry["Y"] * database_entry["confidence"]
                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_Y"] = 1 + database_entry["confidence"]
                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_Z"] = database_entry["Z"] * database_entry["confidence"]
                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_Z"] = 1 + database_entry["confidence"]

                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["X"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_X"]
                    / database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_X"]
                )
                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["Y"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_Y"]
                    / database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_Y"]
                )
                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["Z"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_Z"]
                    / database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_Z"]
                )

            else:
                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_X"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_X"]
                    + database_entry["X"] * database_entry["confidence"]
                )

                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_Y"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_Y"]
                    + database_entry["Y"] * database_entry["confidence"]
                )

                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_Z"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_Z"]
                    + database_entry["Z"] * database_entry["confidence"]
                )

                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_X"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_X"] + database_entry["confidence"]
                )

                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_Y"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_Y"] + database_entry["confidence"]
                )

                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_Z"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_Z"] + database_entry["confidence"]
                )

                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["X"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_X"]
                    / database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_X"]
                )

                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["Y"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_Y"]
                    / database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_Y"]
                )

                database_entries[database_entry["Name"]][database_entry["Corner_name"]]["Z"] = (
                    database_entries[database_entry["Name"]][database_entry["Corner_name"]]["num_Z"]
                    / database_entries[database_entry["Name"]][database_entry["Corner_name"]]["denom_Z"]
                )

        else:
            database_entries[database_entry["Name"]] = {}
    f.close()

with open(SEMANTIC_DATABASE_AVERAGE_FILE, "w") as f:
    json.dump(database_entries, f, indent=4)
    f.close()
