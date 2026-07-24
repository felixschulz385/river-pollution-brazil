devtools::install_github("danicat/read.dbc")

setwd("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis")
library("read.dbc")
library(tidyverse)
library(lubridate)
library(rvest)
library(arrow)
library(sf)
options(timeout=300)

counties = c('AC','AL','AM','AP','BA','CE','DF','ES','GO','MA',
  'MG','MS','MT','PA','PB','PE','PI','PR','RJ','RN',
  'RO','RR','RS','SC','SE','SP','TO')
years = c(2015, 2022)

for(year in years){
    links = paste0("ftp://ftp.datasus.gov.br/dissemin/publicos/SIM/CID10/DORES/DO", counties, year, ".dbc")
    filenames = paste0("data/brazil_mortality/DO", counties, year, ".dbc")
    for (i in 1:length(links)) {
        if(file.exists(filenames[i])) next
        Sys.sleep(1)
        # Download the files using download.file
        download.file(url = links[i], destfile = filenames[i], mode = "wb")
    }
}

###
#
###

death_record_files = list.files("data/brazil_mortality") %>% paste("data/brazil_mortality/", ., sep = "")
death_record_files_years = death_record_files %>% str_sub(-8, -5) %>% as.numeric()
death_record_files = na.omit(death_record_files[death_record_files_years == 2015])

# for (i in 1:length(mortality_files)) {
#     mortality_files[i] %>% read.dbc()
# }

death_record_files %>%
    map_df(read.dbc) %>%
    write_parquet("data/brazil_mortality/death_records_2015.parquet")

# read ods file
# https://www.ibge.gov.br/en/statistics/social/population/18448-estimates-of-resident-population-for-municipalities-and-federation-units.html
# census_data = readxl::read_excel("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/brazil_mortality/POP2021_20230710.xls",
#                                  sheet = "Municípios")

# census_data = census_data %>%
#     select(mun_1 = `...2`, mun_2 = `...3`, pop = `...5`) %>%
#     mutate(mun = map2_chr(mun_1, mun_2, ~paste0(.x, .y, collapse = ""))) %>%
#     mutate(mun = str_sub(mun, 1, 6), across(pop, as.double)) %>%
#     select(-mun_1, -mun_2) %>%
#     drop_na()

# https://sidra.ibge.gov.br/Tabela/9606
census_data = read_csv("/Users/felixschulz/Downloads/tabela9606.csv", skip = 5)

census_data = census_data %>% 
    #filter(str_detect(Idade, "\\d* a \\d* anos")) %>%
    filter(Idade == "0 a 4 anos") %>%
    mutate(population = as.double(Total),
           municipality = str_sub(Município, 1, 6)) %>%
    select(municipality, population) %>%
    drop_na()

census_data %>% write_parquet("data/brazil_mortality/census_data.parquet")


mortality_data = read_parquet("data/brazil_mortality/death_records_2015.parquet") %>% as_tibble()

mortality_data = mortality_data %>%
    select(CODMUNRES, DTNASC, DTOBITO) %>%
    mutate(across(c(DTNASC, DTOBITO), dmy)) %>%
    mutate(age = interval(DTNASC, DTOBITO) / years(1))

# mortality_data %>%
#     filter(age < 5) %>%
#     group_by(CODMUNRES) %>%
#     summarise(n_deaths = n()) %>%
#     ungroup() %>% 
#     ggplot(.) +
#     geom_histogram(aes(x = n_deaths), bins = 50) +
#     scale_x_continuous(limits = c(0, 100))

# mortality_data %>%
#     filter(age < 5) %>%
#     select(CODMUNRES) %>% unique() %>% nrow()

# read.dbc("/Users/felixschulz/Downloads/DOINF22.dbc") %>%
#     as_tibble() %>%
#     select(CODMUNRES) %>% unique() %>% nrow()

# # plot age distribution in deaths
# ggplot(mortality_data, aes(x = age)) +
#     geom_histogram(bins = 50) +
#     scale_x_continuous(limits = c(0, 125)) +
#     theme_bw()
# ggplot(mortality_data %>% filter(CODMUNRES == "110050"), aes(x = age)) +
#     geom_histogram(bins = 50) +
#     scale_x_continuous(limits = c(0, 125)) +
#     theme_bw()

mortality_data = mortality_data %>%
    filter(age <= 5) %>%

    group_by(CODMUNRES) %>%
        summarise(n_deaths = n()) %>%
        ungroup() %>%
    full_join(census_data, by = c("CODMUNRES" = "municipality")) %>%
    mutate(n_deaths = ifelse(is.na(n_deaths), 0, n_deaths)) %>%
    mutate(mortality_rate = n_deaths / population)

mortality_data %>% write_parquet("data/brazil_mortality/mortality_data_u5.parquet")


map = sf::read_sf("/Users/felixschulz/Downloads/gadm41_BRA_2.json")

map = map %>% 
    mutate(CC_2 = CC_2 %>% str_sub(1, 6)) %>%
    st_transform(4326)

map %>% sf::write_sf("data/maps/brazil_municipalities_map.json", driver = "GeoJSON")

##

merged = mortality_data %>% 
    left_join(map, by = c("CODMUNRES" = "CC_2")) %>%
    drop_na() %>%
    st_sf()

merged$mortality_rate %>% summary()

mortality_plot = ggplot() +
  geom_sf(data = merged, aes(fill = mortality_rate)) +
  scale_fill_viridis_c(option = "magma", direction = -1) +
  theme_bw()


mortality_plot %>% ggsave("output/mortality_plot.png", ., width = 10, height = 10, dpi = 300)


###
#
###

vote = read_csv2("data/brazil_vote/votacao_partido_munzona_2022/votacao_partido_munzona_2022_BRASIL.csv")
vote %>% write_parquet("data/brazil_vote/votacao_partido_munzona_2022_BRASIL.parquet")

vote = read_parquet("data/brazil_vote/votacao_partido_munzona_2022_BRASIL.parquet")



dump_files = paste0("data/brazil_misc/postcodes/dump/", list.files("data/brazil_misc/postcodes/dump/"))
for(file in dump_files){
    unzip(file)
}

postcodes_files = paste0("data/brazil_misc/postcodes/dump/", list.files("data/brazil_misc/postcodes/dump/", pattern = ".csv"))

postcodes_files %>% map_df(read_csv, 
                           col_names = c("CEP", "Logradouro", "Bairro", "base", "IDcidade", "IDestado"),
                           col_types = "cccccc") %>% write_parquet("data/brazil_misc/postcodes/postcodes.parquet")

postcodes = read_parquet("data/brazil_misc/postcodes/postcodes.parquet")
cidades = read_csv("data/brazil_misc/postcodes/cities.csv",
                   col_names = c("IDcidade", "cidade", "IDestado"),
                   col_types = "ccc")
states = read_csv("data/brazil_misc/postcodes/states.csv",
                  col_names = c("IDestado", "estado", "estado_abbr"),
                  col_types = "ccc")

# https://www.ibge.gov.br/explica/codigos-dos-municipios.php
html = read_html("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/brazil_misc/postcodes/municipios.html",
                 encoding = "UTF-8")

tmp = html %>% 
    html_elements(".municipio")

states_lookup_table = html %>% 
    html_elements(".uf") %>% 
    map_df(., ~list("estado" = .x %>% html_elements("a") %>% first() %>% html_text(),
                    "estado_code" = .x %>% html_elements("a") %>% last() %>% html_text() %>% str_sub(1,2)))

lookup_table = tmp %>% map_df(., ~ list("name" = .x %>% html_element("a") %>% html_text(),
                                     "ID" = .x %>% html_element(".numero") %>% html_text()))

lookup_table = lookup_table %>%
    mutate(tmp = str_sub(ID, 1, 2)) %>%
    left_join(states_lookup_table, by = c("tmp" = "estado_code")) %>%
    select(-tmp)

postcode_lookup_table = postcodes %>% 
    left_join(cidades) %>% 
    left_join(states) %>%
    left_join(lookup_table, by = c("cidade" = "name", "estado")) %>%
    mutate(CEP = str_sub(CEP, 1, 5)) %>%
    select(CEP, ID) %>%
    group_by(CEP) %>%
    summarise(ID = first(ID))

vote %>%
    filter(CD_ELEICAO == 545) %>%
    select(CD_MUNICIPIO, SG_PARTIDO, QT_VOTOS_NOMINAIS_VALIDOS) %>%
    left_join(postcode_lookup_table, by = c("CD_MUNICIPIO" = "CEP")) %>% View()
    group_by(ID, SG_PARTIDO) %>%
        summarise()

#

#install.packages("basedosdados")
library("basedosdados")
library("tidyverse")
library("arrow")

set_billing_id("master-thesis-412105")

# Para carregar o dado direto no R
query <- bdplyr("br_tse_eleicoes.resultados_candidato_municipio")

df <- bd_collect(query)
df %>% write_parquet("data/brazil_vote/resultados_candidato_municipio.parquet")

vote = read_parquet("data/brazil_vote/resultados_candidato_municipio.parquet")


query <- bdplyr("br_ibge_pnad.microdados_compatibilizados_domicilio")
df <- bd_collect(query)

bd_