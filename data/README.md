# Data (not included)

| Input | Source | Format expected here |
|---|---|---|
| Daily rainfall at gauge stations, 1975–2005 | Iranian provincial agencies (not redistributable) | `stations_predictand.csv`: `date` + one column per station, mm/day, `-99` missing. `ksdsm.io.long_to_wide` converts a long table. `stations_sites.csv`: `site,lat,lon` |
| PERSIANN-CDR daily, 0.25° | NOAA NCEI or CHRS (free) | `python -m ksdsm persiann-download` then `python -m ksdsm persiann-cells` |
| NCEP/NCAR and CanESM2 predictors per grid box | Canadian climate scenarios service (CanESM2 SDSM predictor packages) | `predictors/<site>/ncep/*.dat`, `predictors/<site>/{historical,rcp26,rcp45,rcp85}/*.dat`, one value per line |

The CanESM2 predictor packages were distributed through the Canadian Climate Change Scenarios Network (www.cccsn.ec.gc.ca), later the Canadian Climate Data and Scenarios site. Download the grid box that contains each station or cell and unpack it into the layout above.
