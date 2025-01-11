import json
if __name__ == "__main__":
      file_name_to_content_name = {
            "src/raw_data/buffets.json": "buffets",
            "src/raw_data/coffe_breaks.json": "coffe breaks",
            "src/raw_data/hotel_price.json": "hotel price policy",
            "src/raw_data/wines.json": "winery",
      }
      all_data_json_file = "src/input_data/all_data.json"
      all_data = {}
      for file_name, content_name in file_name_to_content_name.items():
            with open(file_name, "r") as f:
                  all_data[content_name] = json.load(f)
      with open(all_data_json_file, "w") as f:
            json.dump(all_data, f, indent=4)
            
