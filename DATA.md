
# Data directory structure:
Please organize the datasets under the `benchmark/RAW_DATA/` directory as described below. This structure follows the format used in the repository [missing_aware_prompts](https://github.com/YiLunLee/missing_aware_prompts).



## MM-IMDb
[MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb) [(archive.org mirror)](https://archive.org/download/mmimdb)

    IMDB
    ├── images            
    │   ├── 00000005.jpeg 
    │   ├── 00000008.jpeg   
    │   └── ...        
    ├── labels          
    │   ├── 00000005.json 
    │   ├── 00000008.json   
    │   └── ...        
    └── split.json 


Using these command
```bash
# IMDB
mkdir -p benchmark/RAW_DATA/IMDB
cd benchmark/RAW_DATA/IMDB
wget https://archive.org/download/mmimdb/mmimdb.tar.gz
curl -O https://raw.githubusercontent.com/YiLunLee/missing_aware_prompts/refs/heads/main/datasets/mmimdb/split.json
tar -xzvf mmimdb.tar.gz
cd ../../..
python notebook/create_fols_imdb.py
rm -r benchmark/RAW_DATA/IMDB/mmimdb
python notebook/make_arrow_imdb.py
mkdir benchmark/RAW_DATA/IMDB/missing_tables/
mkdir benchmark/RAW_DATA/IMDB/missing_tables_other_tests/
```



## Food101
[UPMC Food-101](https://visiir.isir.upmc.fr/explore) [(Kaggle)](https://www.kaggle.com/datasets/gianmarco96/upmcfood101?select=texts)

    FOOD101
    ├── images            
    │   ├── train                
    │   │   ├── apple_pie
    │   │   │   ├── apple_pie_0.jpg        
    │   │   │   └── ...         
    │   │   ├── baby_back_ribs  
    │   │   │   ├── baby_back_ribs_0.jpg        
    │   │   │   └── ...    
    │   │   └── ...
    │   ├── test                
    │   │   ├── apple_pie
    │   │   │   ├── apple_pie_0.jpg        
    │   │   │   └── ...         
    │   │   ├── baby_back_ribs  
    │   │   │   ├── baby_back_ribs_0.jpg        
    │   │   │   └── ...    
    │   │   └── ...
    ├── texts          
    │   ├── train_titles.csv            
    │   └── test_titles.csv         
    ├── class_idx.json         
    ├── text.json         
    └── split.json

```bash
# Food101
mkdir -p benchmark/RAW_DATA/FOOD101
cd benchmark/RAW_DATA/FOOD101
curl -L -o upmcfood101.zip https://www.kaggle.com/api/v1/datasets/download/gianmarco96/upmcfood101
unzip upmcfood101.zip
curl -O https://raw.githubusercontent.com/YiLunLee/missing_aware_prompts/refs/heads/main/datasets/Food101/class_idx.json
curl -O https://raw.githubusercontent.com/YiLunLee/missing_aware_prompts/refs/heads/main/datasets/Food101/split.json
curl -O https://raw.githubusercontent.com/YiLunLee/missing_aware_prompts/refs/heads/main/datasets/Food101/text.json
cd ../../..
python notebook/make_arrow_food101.py
mkdir benchmark/RAW_DATA/FOOD101/missing_tables_8_classes/
mkdir benchmark/RAW_DATA/FOOD101/missing_tables_other_tests/
```