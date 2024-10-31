# load libraries
library(ggplot2)
library(ggraph)
library(igraph)
library(tidyverse)
library(dplyr)
library(gridExtra)
library(RColorBrewer) 
library(grid)
library(readxl)



##### prepare data for plotting #####

setwd('') # specify filepath to data set
df <- read_xlsx("Prütz_et_al_2024_CDR_side_effects_data_set_v1.0.xlsx",
                sheet = "D1. Literature data")

# remove empty cells in the 'authors' column and filter out excluded rows
df <- df[!is.na(df$authors), ]
df <- subset(df, excluded != 'yes') # keep non excluded rows
df <- subset(df, pub_type == 'Article') # choose between "Article" and/or "Review"

# add origin column for hierarchy plot (center node)
df <- df %>% mutate(origin = '')
df <- df[order(df$category_l0.1, df$category_l1, df$category_l2, df$category_l3),]

# create edgelists from considered columns
root_cat1 <- df %>% 
  select(origin, category_l1) %>% 
  unique %>% 
  rename(from=origin, to=category_l1)

cat1_cat2 <- df %>% 
  select(category_l1, category_l2) %>% 
  unique %>% 
  rename(from=category_l1, to=category_l2)

cat2_cat3 <- df %>% 
  select(category_l2, category_l3) %>% 
  unique %>% 
  rename(from=category_l2, to=category_l3)

df_edgelist=rbind(root_cat1, cat1_cat2, cat2_cat3)



##### prepare and plot effect profiles (figure 3) #####

selected_options <- c('AR', 
                      'BECCS', 
                      'Biochar', 
                      'DACCS', 
                      'EW', 
                      'SCS', 
                      'Unspecified/multiple CDR')

color_palette <- c('AR' = 'red',
                   'BECCS' = 'red',
                   'Biochar' = 'red',
                   'DACCS' = 'red',
                   'EW' = 'red',
                   'SCS' = 'red',
                   'Unspecified/multiple CDR' = 'red',
                   'unselected' = 'gray85')

color_palette2 <- c('TRUE' = 'red',
                   'FALSE' = 'gray85')

plots <- list()
for (selected_option in selected_options) { # create df for following condition statement
  condition_df <- subset(df, cdr_option == selected_option, select = c('cdr_option', 
                                                                       'category_l1', 
                                                                       'category_l2',
                                                                       'category_l3'))
  # specify cdr option - effect relationships
  nodes <- data.frame(all_nodes = unique(c(as.character(df_edgelist$from), 
                                           as.character(df_edgelist$to))), group = 'unselected') 
  nodes$group[nodes$all_nodes %in% condition_df$category_l1] <- selected_option
  nodes$group[nodes$all_nodes %in% condition_df$category_l2] <- selected_option
  nodes$group[nodes$all_nodes %in% condition_df$category_l3] <- selected_option
  
  # specify which edges to highlight with color
  filter <- subset(nodes, group != 'unselected')
  df_edgelist$edge_c <- FALSE
  df_edgelist$edge_c <- ifelse(df_edgelist$to %in% filter$all_nodes, TRUE, FALSE)
  
  # create hierarchy_graph from edgelist
  hierarchy_graph <- graph_from_data_frame(df_edgelist, vertices = nodes)

  # define circle for graph and control width of the slices
  radius <- 1.05
  radius2 <- 1.02
  circle_1 <- data.frame(
    x = c(0, radius * cos(seq(pi/2, 0.85, length.out = 100)), 0),
    y = c(0, radius * sin(seq(pi/2, 0.85, length.out = 100)), 0))
  circle_2 <- data.frame(
    x = c(0, -radius * cos(seq(pi/2, -0.25, length.out = 100)), 0),
    y = c(0, radius * sin(seq(pi/2, -0.25, length.out = 100)), 0))
  circle_3 <- data.frame(
    x = c(0, radius * cos(seq(pi/1.08648, -0.85, length.out = 100)), 0),
    y = c(0, -radius * sin(seq(pi/1.08648, -0.85, length.out = 100)), 0))
  circle_4 <- data.frame(
    x = c(0, radius2 * cos(seq(0, 2*pi, length.out = 100))),
    y = c(0, -radius2 * sin(seq(0, 2*pi, length.out = 100))))
  
  plot <- ggraph(hierarchy_graph, 'igraph', algorithm = 'tree', circular = TRUE) + 
    geom_polygon(data = circle_1, aes(x, y), fill = 'blue', alpha = 0.6) +
    geom_polygon(data = circle_2, aes(x, y), fill = 'orange', alpha = 0.6) +
    geom_polygon(data = circle_3, aes(x, y), fill = 'green', alpha = 0.6) +
    geom_polygon(data = circle_4, aes(x, y), fill = 'white', alpha = 1) +
    geom_edge_diagonal2(aes(colour = edge_c), alpha = 1, width = 0.3) +
    scale_edge_color_manual(values = color_palette2) +
    geom_node_point(aes(color = group), size = 0.8) +
    scale_color_manual(values = color_palette) +
    theme_void() +
    coord_fixed() + 
    expand_limits(x = c(-1.25, 1.25), y = c(-1.3, 1.3)) +
    theme(legend.position = 'none',
          plot.title = element_text(size = 11, hjust = 0.5, vjust=1,  margin = margin(0, 0, -20, 0)),
          plot.margin = margin(-25, -20, -25, -20)) +
    labs(title = paste(selected_option))
  plots[[selected_option]] <- plot}

grid.arrange(grobs = plots, ncol = 3, vjust=0.1) # create subplots for each cdr option



##### prepare and plot igraph tree based on hierarchy_graph (figure 2) #####

# abbreviate words for better readability of plot
unique_original <- unique(c(df_edgelist$from, df_edgelist$to))
abbreviate_multiword <- function(names, minlength_first = x, minlength_following = x) {
  words <- str_split(names, '\\s+')
  abbrev_words <- lapply(words, function(word) {
    if (length(word) == 1) {
      ifelse(nchar(word) <= minlength_first, word, str_sub(word, 1, minlength_first))
    } else {
      abbrev <- c(str_sub(word[1], 1, minlength_first), sapply(word[-1], function(w) {
        ifelse(nchar(w) <= minlength_following, w, str_sub(w, 1, minlength_following))
      }))
      abbrev}})
  abbrev_entries <- sapply(abbrev_words, paste, collapse = ' ')
  return(abbrev_entries)}

# create abbreviations table
abbreviations <- abbreviate_multiword(unique_original, 
                                      minlength_first = 15, 
                                      minlength_following = 15) # choose length for first and following words
abbreviations_table <- data.frame(original = unique_original,
                                  abbreviation = abbreviations)

# update abbreviation table to make duplicates unique
duplicates <- duplicated(abbreviations_table$abbreviation)
counter <- ave(duplicates, abbreviations_table$abbreviation, FUN = cumsum)
abbreviations_table <- within(abbreviations_table, {
  unique <- ifelse(duplicates, paste(abbreviation, counter, sep = '.'), abbreviation)})

################################################################################
# alternative: abbreviate based on customized abbreviation list (optional step)
# specify custom abbreviations list based on "abbreviations_table" of previous step
# required input: csv file with original names column and desired abbreviation column
abbreviation_list <- read_csv('abbreviations_list.csv') # specify custom list here or use provided csv file
empty_row <- data.frame(text = '', unique_abbrev = '', taxo_number = '', abbrev_num = '') 
abbreviation_list <- rbind(empty_row, abbreviation_list)
abbreviations_table = cbind(abbreviations_table, abbreviation_list) # double check that all entries match
################################################################################

# abbreviate edgelist based on updated abbreviation table (choose abbreviation option)
abbrev_dict <- setNames(abbreviations_table$taxo_number, # choose $unique_abbrev, $taxo_number or $abbrev_num
                        abbreviations_table$original) 
df_edgelist$from <- abbrev_dict[match(df_edgelist$from, names(abbrev_dict))]
df_edgelist$to <- abbrev_dict[match(df_edgelist$to, names(abbrev_dict))]

# create hierarchy_graph from edgelist
hierarchy_graph <- graph_from_data_frame(df_edgelist)

# define circle radius for graph
radius <- 1.03
radius2 <- 1.02

ggraph(hierarchy_graph, 
       'igraph', 
      algorithm='tree',
       circular=TRUE) + 
  geom_polygon(data = circle_1, aes(x, y), fill = 'blue', alpha = 0.6) +
  geom_polygon(data = circle_2, aes(x, y), fill = 'orange', alpha = 0.6) +
  geom_polygon(data = circle_3, aes(x, y), fill = 'green', alpha = 0.6) +
  geom_polygon(data = circle_4, aes(x, y), fill = 'white', alpha = 1) +
  geom_edge_diagonal(colour='grey70', alpha=1, width=0.22) +
  geom_node_point(color='grey60', size = 1.3) +
  geom_node_text(aes(x=x*1.07, 
                     y=y*1.07, 
                     label=name, 
                     angle=-((-node_angle(x, y)+90)%%180)+90,
                     hjust=ifelse(rank(x) <= length(x)/2, 1, 0)), 
                 size=3.1, 
                 color='black') +
  theme_void() +
  coord_fixed() + 
  expand_limits(x = c(-1.135, 1.135), y = c(-1.12, 1.12))

# info: the legend for figure 1, 2 and S2 was added separately to the created plots