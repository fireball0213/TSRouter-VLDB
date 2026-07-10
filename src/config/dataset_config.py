Med_long_Fast_datasets = (
        "SZ_TAXI/15T "
        "bizitobs_application "
        "bizitobs_l2c/5T bizitobs_l2c/H "
        "ett1/15T ett1/H ett2/15T ett2/H "
        "jena_weather/H "
)
Short_Fast_datasets = (
        "M_DENSE/D "
        "bizitobs_application "
        "bizitobs_l2c/5T bizitobs_l2c/H  "
        "electricity/W "
        "ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
        "jena_weather/H jena_weather/D "
        "hierarchical_sales/D hierarchical_sales/W "
        "hospital "
        "saugeenday/D saugeenday/M saugeenday/W "
        "solar/D solar/W "
        "SZ_TAXI/15T SZ_TAXI/H "
        "m4_weekly "
        "m4_hourly "
        "covid_deaths "
        "us_births/D us_births/M us_births/W "
        "LOOP_SEATTLE/D "
)
ALL_Fast_DATASETS = [
    'bizitobs_application/10S/long',
    'bizitobs_application/10S/medium',
    'bizitobs_application/10S/short',
    'bizitobs_l2c/5T/long',
    'bizitobs_l2c/5T/medium',
    'bizitobs_l2c/5T/short',
    'bizitobs_l2c/H/long',
    'bizitobs_l2c/H/medium',
    'bizitobs_l2c/H/short',
    'covid_deaths/D/short',
    'electricity/W/short',
    'ett1/15T/long',
    'ett1/15T/medium',
    'ett1/15T/short',
    'ett1/D/short',
    'ett1/H/long',
    'ett1/H/medium',
    'ett1/H/short',
    'ett1/W/short',
    'ett2/15T/long',
    'ett2/15T/medium',
    'ett2/15T/short',
    'ett2/D/short',
    'ett2/H/long',
    'ett2/H/medium',
    'ett2/H/short',
    'ett2/W/short',
    'hierarchical_sales/D/short',
    'hierarchical_sales/W/short',
    'hospital/M/short',
    'jena_weather/D/short',
    'jena_weather/H/long',
    'jena_weather/H/medium',
    'jena_weather/H/short',
    'loop_seattle/D/short',
    'm4_hourly/H/short',
    'm4_weekly/W/short',
    'm_dense/D/short',
    'saugeen/D/short',
    'saugeen/M/short',
    'saugeen/W/short',
    'solar/D/short',
    'solar/W/short',
    'sz_taxi/15T/long',
    'sz_taxi/15T/medium',
    'sz_taxi/15T/short',
    'sz_taxi/H/short',
    'us_births/D/short',
    'us_births/M/short',
    'us_births/W/short'
]
Short_datasets = (
        "M_DENSE/H M_DENSE/D "
        "bitbrains_rnd/5T "   
        "bitbrains_rnd/H "
        "bizitobs_application "
        "bizitobs_l2c/5T bizitobs_l2c/H  "
        "bizitobs_service "
        "electricity/15T "  
        "electricity/H "   
         "electricity/D electricity/W "
        #
        "ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
        "jena_weather/10T jena_weather/H jena_weather/D "
        "hierarchical_sales/D hierarchical_sales/W "
        "hospital "
        "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
        "restaurant "
        "saugeenday/D saugeenday/M saugeenday/W "
        "solar/10T "   
        "solar/H solar/D solar/W "
        "SZ_TAXI/15T SZ_TAXI/H "
        "m4_quarterly m4_yearly m4_monthly "  
        "m4_weekly m4_daily m4_hourly "

        "temperature_rain_with_missing "  
        "car_parts_with_missing "
        "covid_deaths "
        "us_births/D us_births/M us_births/W "
        "bitbrains_fast_storage/5T bitbrains_fast_storage/H "  
        "LOOP_SEATTLE/5T "  
        "LOOP_SEATTLE/H "
        "LOOP_SEATTLE/D "
)

Med_long_datasets = (
        "M_DENSE/H "
        "SZ_TAXI/15T "
        "bitbrains_rnd/5T "
        "bizitobs_application "
        "bizitobs_service "
        "bizitobs_l2c/5T bizitobs_l2c/H "
        "electricity/15T "
        "electricity/H "
        "ett1/15T ett1/H ett2/15T ett2/H "
        "jena_weather/10T jena_weather/H "
        "kdd_cup_2018_with_missing/H "
        "solar/10T solar/H "
        "bitbrains_fast_storage/5T "
        "LOOP_SEATTLE/5T "
        "LOOP_SEATTLE/H "
)

                   
ALL_DATASETS = [
    'bitbrains_fast_storage/5T/long',
    'bitbrains_fast_storage/5T/medium',
    'bitbrains_fast_storage/5T/short',
    'bitbrains_fast_storage/H/short',
    'bitbrains_rnd/5T/long',
    'bitbrains_rnd/5T/medium',
    'bitbrains_rnd/5T/short',
    'bitbrains_rnd/H/short',
    'bizitobs_application/10S/long',
    'bizitobs_application/10S/medium',
    'bizitobs_application/10S/short',
    'bizitobs_l2c/5T/long',
    'bizitobs_l2c/5T/medium',
    'bizitobs_l2c/5T/short',
    'bizitobs_l2c/H/long',
    'bizitobs_l2c/H/medium',
    'bizitobs_l2c/H/short',
    'bizitobs_service/10S/long',
    'bizitobs_service/10S/medium',
    'bizitobs_service/10S/short',
    'car_parts/M/short',
    'covid_deaths/D/short',
    'electricity/15T/long',
    'electricity/15T/medium',
    'electricity/15T/short',
    'electricity/D/short',
    'electricity/H/long',
    'electricity/H/medium',
    'electricity/H/short',
    'electricity/W/short',
    'ett1/15T/long',
    'ett1/15T/medium',
    'ett1/15T/short',
    'ett1/D/short',
    'ett1/H/long',
    'ett1/H/medium',
    'ett1/H/short',
    'ett1/W/short',
    'ett2/15T/long',
    'ett2/15T/medium',
    'ett2/15T/short',
    'ett2/D/short',
    'ett2/H/long',
    'ett2/H/medium',
    'ett2/H/short',
    'ett2/W/short',
    'hierarchical_sales/D/short',
    'hierarchical_sales/W/short',
    'hospital/M/short',
    'jena_weather/10T/long',
    'jena_weather/10T/medium',
    'jena_weather/10T/short',
    'jena_weather/D/short',
    'jena_weather/H/long',
    'jena_weather/H/medium',
    'jena_weather/H/short',
    'kdd_cup_2018/D/short',
    'kdd_cup_2018/H/long',
    'kdd_cup_2018/H/medium',
    'kdd_cup_2018/H/short',
    'loop_seattle/5T/long',
    'loop_seattle/5T/medium',
    'loop_seattle/5T/short',
    'loop_seattle/D/short',
    'loop_seattle/H/long',
    'loop_seattle/H/medium',
    'loop_seattle/H/short',
    'm4_daily/D/short',
    'm4_hourly/H/short',
    'm4_monthly/M/short',
    'm4_quarterly/Q/short',
    'm4_weekly/W/short',
    'm4_yearly/A/short',
    'm_dense/D/short',
    'm_dense/H/long',
    'm_dense/H/medium',
    'm_dense/H/short',
    'restaurant/D/short',
    'saugeen/D/short',
    'saugeen/M/short',
    'saugeen/W/short',
    'solar/10T/long',
    'solar/10T/medium',
    'solar/10T/short',
    'solar/D/short',
    'solar/H/long',
    'solar/H/medium',
    'solar/H/short',
    'solar/W/short',
    'sz_taxi/15T/long',
    'sz_taxi/15T/medium',
    'sz_taxi/15T/short',
    'sz_taxi/H/short',
    'temperature_rain/D/short',
    'us_births/D/short',
    'us_births/M/short',
    'us_births/W/short'
]
Short_uni_datasets = (
        "M_DENSE/H M_DENSE/D "
        # "bitbrains_rnd/5T bitbrains_rnd/H "
        # "bizitobs_application "
        # "bizitobs_l2c/5T bizitobs_l2c/H  "
        # "bizitobs_service "
                                                                        

        # "ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
        # "jena_weather/10T jena_weather/H jena_weather/D "
        "hierarchical_sales/D hierarchical_sales/W "
        "hospital "
        "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
        "restaurant "
        "saugeenday/D saugeenday/M saugeenday/W "
        # "solar/10T solar/H "
        "solar/D solar/W "
        "SZ_TAXI/15T SZ_TAXI/H "
        "m4_hourly m4_weekly "
                                                           

                                            
        "car_parts_with_missing "
        "covid_deaths "
        "us_births/D us_births/M us_births/W "
        # "bitbrains_fast_storage/H "
                                        
                                                            
)

Med_uni_datasets = (
        "M_DENSE/H "
        "SZ_TAXI/15T "
        # "bitbrains_rnd/5T "
        # "bizitobs_application "
        # "bizitobs_service "
        # "bizitobs_l2c/5T bizitobs_l2c/H "
        "electricity/15T electricity/H "
        # "ett1/15T ett1/H ett2/15T ett2/H "
        # "jena_weather/10T jena_weather/H "
        "kdd_cup_2018_with_missing/H "
        "solar/10T solar/H "
        # "bitbrains_fast_storage/5T "
        "LOOP_SEATTLE/5T LOOP_SEATTLE/H "
)

