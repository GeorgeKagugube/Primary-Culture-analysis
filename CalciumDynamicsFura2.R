## Clear the workspce here 
rm(list = ls())

# Set the working directory here
setwd('/Users/gwk/Desktop/PhD/Data/PhD_data/250226')

# Load required packages
library(ggplot2)
library(ggsignif)
#library(readr)
library(gridExtra)
library(tidyverse)
library(ggpubr)

# Load the data
df <- read_csv("fura-2_AM_Rhodamine123.csv", show_col_types = F)

## Quick exploration
head(df)
tail(df)
str(df)

# Ensure proper factor levels for consistent ordering
#data$Genotype <- factor(data$Genotype, levels = c("WT", "Het", "Hom"))
df$Genotype <- factor(df$Genotype, levels = c("WT","Hom"))
df$Stimulant <- factor(df$Stimulant, levels = c('Glutamate','ATP', 'KCl'))
df$Celltype <- factor(df$Celltype, levels = c("Neurone", "Astrocytes"))

# Re-arrange the categorical groups here
df$Genotype <- ordered(df$Genotype,
                           levels = c('WT', 'Hom'))
df$Stimulant <- ordered(df$Stimulant,
                          levels = c('Glutamate', 'ATP', 'KCl'))

## Check the names of the columns in the dataframe here
names(df)
attach(df)

## Filter and visualise data here with statistics
data_selection <- function(dataFrame, column = Stimulant, selection = 'Glutamate') {
  new_data_frame <- dataFrame %>%
    filter(column == selection)
  
  # Return the newdataframe here 
  return(new_data_frame)
}

## New dataframe to work with here
newdf <- data_selection(df)

# Run statistics here 
## qqplot 
ggqqplot(newdf$Peak)

## Test for normality of dataset
# A pvalue > 0.05 == normal distribution,
# A pvalue < 0.05 ==> other distribution
shapiro.test(newdf$AUC)

## For normal distrobution and two sample means, a test is used
t.test(Repolarisation_Slope ~ Genotype, data = newdf)

## Test of two independent variables, with unknown variance and non-normallity
wilcox.test(Repolarisation_Slope ~ Genotype, data = newdf, exact = FALSE)

## One-way anova
res_ov <- aov(Repolarisation_Slope ~ Genotype, data = df)
summary(res_ov)

# Create a box-plot
ggboxplot(
  newdf, x = "Genotype", y = "Repolarisation_Slope", facet.by = 'Celltype',
  ylab = "Repolarisation_Slope (s-1)", xlab = "Genotype", add = "jitter", width = 0.5)
