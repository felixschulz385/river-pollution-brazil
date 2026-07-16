###
# Test against DETER cloud cover data
###

cloud_cover <- arrow::read_parquet("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/cloud_cover/cloud_cover_DETER.parquet") %>% 
    mutate(CC_2r = as.character(CC_2r))
# has 803 municipalities per year between 2005 and 2017

DETER_test_data <- inner_join(deforestation %>% filter(year >= 2004, CC_2r %in% legal_amazon) %>% mutate(deforestation_rate = deforestation / total), cloud_cover, by = c("CC_2r", "year")) %>%
    arrange(CC_2r, year) %>%
    # create lagged variants of the cloud cover data
    group_by(CC_2r) %>%
    mutate(cloud_cover.x.l1 = dplyr::lag(cloud_cover.x, 1),
           cloud_cover.y.l1 = dplyr::lag(cloud_cover.y, 1)) %>%
    ungroup()


# plot DETER against ESA cloud cover data
ggplot(DETER_test_data) +
    geom_point(aes(x = cloud_cover.x, y = cloud_cover.y), alpha = .3) +
    geom_smooth(aes(x = cloud_cover.x, y = cloud_cover.y), method = "lm", se = FALSE, color = "purple", linetype = "dashed") +
    theme_bw()

# plot DETER against ESA cloud cover data
ggplot(DETER_test_data) +
    geom_point(aes(x = deforestation_rate, y = cloud_cover.x.l1), alpha = .3)

cor.test(DETER_test_data$cloud_cover.x, DETER_test_data$cloud_cover.y)

felm(deforestation_rate ~ cloud_cover.x | CC_2r + year, data = DETER_test_data) %>% summary()

felm(deforestation_rate ~ cloud_cover.y | CC_2r + year, data = DETER_test_data) %>% summary()