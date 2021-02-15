#################################################################################################
# 01. DATASET CREATION

# The aim of this script is to create a dataset with the following information:
#   - Name of the article
#   - Content of the article
#   - Category of the article
#################################################################################################
# Installs
#install.packages("readtext", dependencies=T)

# Imports
library(readtext)

# Cleaning environment data
rm(list = ls())

# Working directory
#setwd('')

# Path definition of the news archives
path <- 'data/bbc-fulltext/bbc'

# List with the 5 categories
list_categories <- list.files(path=path)

print(list_categories)

# à terminer. c'était pour m'assurer que le csv produit par dataset_creation.py était le même que celui-ci (de l'auteur de l'article) mais pas le temps.