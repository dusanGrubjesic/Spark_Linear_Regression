---
title: "README"
author: "Dusan Grubjesic email: grubjesic.dusan@gmail.com"
date: "August 11, 2015"
output: html_document
---

# Click rate prediction algorithm

This is click rate prediction algorithm using [spark](http://spark.apache.org/), writen in python api of spark: [pyspark](http://spark.apache.org/docs/latest/api/python/index.html).

## Data

Data was taken from [Criteo Labs](http://labs.criteo.com/) and is sample of Kaggle Display Advertising Challenge Dataset.
It can be downloaded after you accept the agreement
[http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/](http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/).

It is structured as lines of observations where first is click or no click(1,0) and rest is features

## Before start

You must have installed apache spark and python.
Also you have to change location of sample in ClickRate.py to where you downloaded it and spark context if you want to change from local to cluster. Sh file is only used for simpler starting and if you want to use it you have to change to your settings.

<sub>I have apache spark pre-bult with hadoop 2.6, python 3.4 and numpy package installed</sub>

## Process

1) Sample is first parsed and loaded in context.
2) Transformed so it can be used in logistic regression
3) Model created from train data
4) Set of log loss validations
5) Iterations of logistic regressions for best hyperparamaters

<sub>additional explanations are in code</sub>

