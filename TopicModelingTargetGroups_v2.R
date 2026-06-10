#State of the Union speeches.
#Data and code inspiration from http://programminghistorian.github.io/ph-submissions/lessons/published/basic-text-processing-in-r

#Install Necessary Packages (just need to do this once)
#install.packages("tidyverse")
#install.packages("tokenizers")
#install.packages("quanteda")
#Load package libraries
library(tidyverse)
library(tokenizers)
library(quanteda)
library(quanteda.textplots)
#New library you will need:LDA and STM 
#install.packages("stm")
library(stm)
#install.packages("seededlda")
library(seededlda)
#Set working directory.  To find your working directory go to Session --> Set Working Directory --> Choose Directory
setwd("C:/Users/ariel/Downloads/UCSD/CSS 206/Project 1/HateSpeech")
#Load data for speeches
metadata <- read_csv("df_toptargets.csv")
#Look at data
metadata

metadata$target = factor(metadata$target)


# gemini code - told it to make my code by faster

# Process data with aggressive cleaning parameters
# This strips out numbers and punctuation *before* building the vocabulary list
temp <- textProcessor(
  documents = metadata$text,
  metadata = metadata,
  lowercase = TRUE,
  removestopwords = TRUE,
  removepunctuation = TRUE,
  removenumbers = TRUE,
  stem = TRUE
)

# CRITICAL SPEEDUP STEP: Prune the sparse vocabulary
# lower.thresh = 15 removes words that appear less than 15 times in the whole dataset.
# For ~20k posts, this will drastically cut processing time while retaining core topics.
out <- prepDocuments(
  temp$documents, 
  temp$vocab, 
  temp$meta, 
  lower.thresh = 15
)

# Run STM with Optimized Settings
my_k = 10  # Note: K=3 is very low for 10 target groups. Let's aim closer to 10 to 20 later.

model.stm <- stm(
  documents = out$documents, 
  vocab = out$vocab, 
  K = my_k, 
  prevalence = ~ target + speech_type,
  data = out$meta, 
  init.type = "Spectral", # MUCH faster convergence than random LDA init
  verbose = TRUE          # Set to TRUE so you can monitor progress live
)

#STM
#install.packages("tm")
#install.packages("SnowballC")

#Process the data to put it in STM format.  Textprocessor automatically does preprocessing
#temp<-textProcessor(documents=metadata$text,metadata=metadata)
#prepDocuments removes words/docs that are now empty after preprocessing
#out <- prepDocuments(temp$documents, temp$vocab, temp$meta)

#Let's try to distinguish between topics that are spoken/written and by year

#This takes a bit. You'd want to remove max.em.its -- this is just to make it shorter!
#Here we are using prevalence covariate sotu_type and year
#my_k = 3
#model.stm <- stm(out$documents, out$vocab, K = my_k, prevalence = ~target + speech_type,data = out$meta, max.em.its = 10) 
#model.stm <- stm(out$documents, out$vocab, K = my_k, prevalence = ~target + speech_type,data = out$meta) 

#Find most probable words in each topic
labelTopics(model.stm)
labelTopics(model.stm, 2)
labelTopics(model.stm, 9)

#And most common topics
plot(model.stm, n=5)
plot(model.stm, n=5, labeltype="frex")

# for a topic Get representative documents (helpful)
findThoughts(model.stm, texts=out$meta$text, n=3)
# for a topic Get representative speech type (somewhat helpful)
findThoughts(model.stm, texts=out$meta$speech_type, n=3)
# for a topic Get representative target type(s) (helpful)
findThoughts(model.stm, texts=out$meta$target, n=5)

findThoughts(model.stm, texts=out$meta$target, topics=1, n=5)
findThoughts(model.stm, texts=out$meta$target, topics=2, n=5)
findThoughts(model.stm, texts=out$meta$target, topics=3, n=5)
findThoughts(model.stm, texts=out$meta$target, topics=4, n=5)
findThoughts(model.stm, texts=out$meta$target, topics=5, n=5)
findThoughts(model.stm, texts=out$meta$target, topics=6, n=5)
findThoughts(model.stm, texts=out$meta$target, topics=7, n=5)
findThoughts(model.stm, texts=out$meta$target, topics=8, n=5)
findThoughts(model.stm, texts=out$meta$target, topics=9, n=5)
findThoughts(model.stm, texts=out$meta$target, topics=10, n=5)


#Estimate relationship between type, year, and topics
model.stm.ee <- estimateEffect(1:10 ~target + speech_type, model.stm, meta = out$meta)
plot(model.stm.ee, "speech_type")
   # the below either dont work or result in increasingly more unreadable plots
plot(model.stm.ee, "target")
plot(model.stm.ee, "text")
plot(model.stm.ee, "speech_type", method="difference", cov.value1="text", cov.value2="target")








############################
#Optional: Using a content covariate
############################

#Let's just look at speeches made by Democratic and Republican presidents after 1950.
meta_sub <- metadata[metadata$party%in%c("Democratic", "Republican") & metadata$year>1950,]
#Process the data to put it in STM format.  Textprocessor automatically does preprocessing
temp<-textProcessor(documents=meta_sub$text,metadata=meta_sub)
#prepDocuments removes words/docs that are now empty after preprocessing
out <- prepDocuments(temp$documents, temp$vocab, temp$meta)

model.stm.c <- stm(out$documents, out$vocab, K = 20, content = ~party,
                   data = out$meta, max.em.its = 10) 
sageLabels(model.stm.c)

plot(model.stm.c, type="perspectives", topics=4)

findThoughts(model.stm.c, texts=out$meta$president, topics=4, n=5)
plot(model.stm.c, type="perspectives", topics=4)
