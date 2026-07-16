library(tidyverse)
library(lfe)
library(texreg)
library(arrow)

setwd("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis")

analysis_panel <- read_csv("data/analysis/stations_panel.csv")
analysis_bins <- read_csv("data/analysis/stations_bins.csv")

dep_vars <- c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates")
dep_vars_table <- c("pH", "Turbidity", "\\makecell{Biochemical\\\\Oxygen\\\\Demand}", "\\makecell{Dissolved\\\\Oxygen}", "\\makecell{Total\\\\Residue}", "\\makecell{Total\\\\Nitrogen}", "Nitrates")


sensor_data <- read_feather("data/water_quality/quality_indicators.feather", c("pH", "Turbidez", "DBO", "OD", "SolTotais", "FosforoTotal", "NitrogenioAmoniacal", "Nitratos", "ColiformesTermotolerantes", "EstacaoCodigo", "date"))

#colnames(read_feather("data/water_quality/quality_indicators.feather"))

sensor_data <- sensor_data %>%
    # rename columns
    rename(
        pH = pH,
        turbidity = Turbidez,
        biochem_oxygen_demand = DBO,
        dissolved_oxygen = OD,
        total_residue = SolTotais,
        total_nitrogen = NitrogenioAmoniacal,
        nitrates = Nitratos,
        phosphorus = FosforoTotal,
        coliforms = ColiformesTermotolerantes,
        station = EstacaoCodigo,
        date = date
    ) %>%
    mutate(year = lubridate::year(date)) %>%
    # convert to numeric (using "," as decimal separator)
    mutate(across(pH:coliforms, ~ as.numeric(str_replace(., ",", ".")))) %>%
    # remove 1.5 IQR outliers
    mutate(across(pH:coliforms, ~ ifelse(. > (median(., na.rm = TRUE) + 1.5 * (quantile(., 0.75, na.rm = TRUE) - quantile(., 0.25, na.rm = TRUE))), NA, .)))
    #mutate(across(pH:coliforms, ~ ifelse(. > quantile(., 0.99, na.rm = TRUE), NA, .)))

# check number of observations per station/year
sensor_data %>%
    group_by(EstacaoCodigo, year = lubridate::year(date)) %>%
    summarize(n = n()) %>%
    ggplot(aes(x = n)) +
    geom_histogram(bins = 30) +
    theme_minimal()

# filter stations with at least 4 observations in year
sensor_data_subset <- sensor_data %>%
    group_by(station, year) %>%
    mutate(n = n()) %>%
    ungroup() %>%
    filter(n >= 4) %>%
    select(-n)

# check variable distribution
sensor_data_subset %>%
    select(pH, turbidity, biochem_oxygen_demand, dissolved_oxygen, total_residue, total_nitrogen, nitrates, phosphorus, coliforms) %>%
    summary()

sensor_data_subset %>%
    select(pH, turbidity, biochem_oxygen_demand, dissolved_oxygen, total_residue, total_nitrogen, nitrates, phosphorus, coliforms) %>%
    gather(key = "variable", value = "value") %>%
    ggplot(aes(x = value)) +
    geom_histogram(bins = 30) +
    facet_wrap(~variable, scales = "free") +
    theme_minimal()
# pH is normally distributed
# biochem_oxygen_demand, dissolved_oxygen, total_residue, total_nitrogen, nitrates are right-skewed count data

# check missing values
sensor_data_subset %>%
    summarize(across(everything(), ~ sum(is.na(.)) / n())) %>%
    gather(key = "variable", value = "missing") %>%
    ggplot(aes(x = variable, y = missing)) +
    geom_col() +
    theme_minimal()

# check 0 values
sensor_data_subset %>%
    summarize(across(everything(), ~ sum(. == 0, na.rm = TRUE) / n())) %>%
    gather(key = "variable", value = "zero") %>%
    ggplot(aes(x = variable, y = zero)) +
    geom_col() +
    theme_minimal()

## Simplify land cover
land_cover_stations <- read_parquet("data/land_cover/land_cover_stations.parquet", as_data_frame = FALSE)
land_cover_stations$bins %>% levels()

land_cover_stations <- land_cover_stations %>%
    group_by(station, year) %>%
    summarise(across(forest:total, ~ sum(., na.rm = TRUE))) %>%
    ungroup()

land_cover_stations %>%
    write_parquet("data/land_cover/land_cover_stations_total.parquet")


## Read simplified land cover
land_cover_stations <- read_parquet("data/land_cover/land_cover_stations_total.parquet", as_data_frame = TRUE)

land_cover_stations <- land_cover_stations %>%
    mutate(across(forest:total, ~ . / total, .names = "s_{.col}"))

# plot distribution of land cover shares
land_cover_stations %>%
    gather(key = "variable", value = "value", s_forest:s_total) %>%
    ggplot(aes(x = log(value + 0.1))) +
    geom_histogram(bins = 30) +
    facet_wrap(~variable, scales = "free") +
    theme_minimal()

land_cover_stations %>%
    reframe(mining = s_mining > 0) %>%
    summary()

analysis_panel <- left_join(
    sensor_data_subset,
    land_cover_stations,
    by = c("station", "year")
    )
    
test_mining_demeaned <- analysis_panel %>%
    group_by(station) %>%
    mutate(
        pH = pH - mean(pH, na.rm = TRUE),
        s_mining = s_mining - mean(s_mining, na.rm = TRUE)
        ) %>%
    ungroup() %>%
    group_by(year) %>%
    mutate(
        pH = pH - mean(pH, na.rm = TRUE),
        s_mining = s_mining - mean(s_mining, na.rm = TRUE)
    ) %>%
    ungroup()


ggplot(test_mining_demeaned, aes(x = s_mining, y = pH)) +
    geom_point() +
    geom_smooth(method = "lm") +
    theme_minimal()

## data is super sparse, maybe look at opening within distance?

land_cover_stations <- read_parquet("data/land_cover/land_cover_stations.parquet")
land_cover_stations %>% 
    filter(bins <= "[50000.0,100000.0)") %>%
    group_by(station, year) %>%
    summarise(across(forest:total, ~ sum(., na.rm = TRUE))) %>%
    ungroup() %>%
    mutate(across(forest:total, ~ . / total, .names = "s_{.col}")) %>%
    write_parquet("data/land_cover/land_cover_stations_100km.parquet")

land_cover_stations <- read_parquet("data/land_cover/land_cover_stations_100km.parquet", as_data_frame = TRUE)

land_cover_stations <- land_cover_stations %>%
    group_by(station) %>%
    mutate(
        mine_ = min(year[s_mining > 0]),
        mine_ = ifelse(is.infinite(mine_), 0, mine_)
        ) %>%
    ungroup()

land_cover_stations %>%
    mutate(mine_ = s_mining > 0) %>%
    group_by(year) %>%
    summarise(mine = sum(mine_, na.rm = TRUE)) %>%
    ggplot(aes(x = year, y = mine)) +
    geom_line() +
    theme_minimal()

analysis_panel <- left_join(
    sensor_data_subset,
    land_cover_stations,
    by = c("station", "year")
    )
    

felm(pH ~ mine_ | station + year, data = analysis_panel) %>% summary()

# summarize the results
summary(example_attgt)


colnames(analysis_panel) %>% paste0(collapse = '", "')

analysis_variables <- c('pH', 'turbidity', 'biochem_oxygen_demand', 'dissolved_oxygen', 'total_residue', 'total_nitrogen', 'nitrates', 'phosphorus', 'coliforms')

# prepare analysis grid
analysis_results <- tibble(
    dep_var = analysis_variables
)

analysis_results <- analysis_results %>%
    mutate(
        formula = map(dep_var, ~ as.formula(paste0(.x, " ~ s_pasture + s_agriculture + s_urban + s_mining | station + year"))),
        model = map2(formula, dep_var, ~ felm(.x, data = analysis_panel))
    )

screenreg(analysis_results$model, custom.model.names = analysis_results$dep_var)

texreg(analysis_results$model,
    file = "output/tables/reg_stations_deforestation.tex",
    custom.model.names = dep_vars_table,
    custom.coef.names = c("Deforestation"),
    stars = c(0.01, 0.05, 0.1),
    caption = "Regression Results: Aggregate Deforestation and Pollution",
    label = "tbl-reg-deforestation-sensors",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.note = "\\item \\textit{Notes:} Unbalanced panels of all available data post 2000. Models include station and year fixed effects. Standard errors are clustered at the watershed and watershed-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = 0.8,
    dcolumn = TRUE
)