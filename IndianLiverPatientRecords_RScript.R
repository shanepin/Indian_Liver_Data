library(tidyverse)
library(reshape2)
library(caret)


# Load data set ###############################################################

wd <- getwd()
file <- "indian_liver_patient.csv"
patients <- read.csv(paste(wd, file, sep = "/"))

# Summary of data set
summary(patients)

# Cleaning Data Set and Exploratory Analysis ###################################

# Explore Outliers in Dataset ##########
rowNum <- c(which.max(patients$Alkaline_Phosphotase),
            which.max(patients$Alamine_Aminotransferase),
            which.max(patients$Aspartate_Aminotransferase))
patients[rowNum,]


# Correlation between Features #########
round(cor(patients[sapply(patients, is.numeric)]),3) %>%
  melt() %>%
  ggplot(aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  labs(
    title = "Correlation Between Attributes"
  ) +
  theme(
    axis.text.x = element_text(angle = 90)
  )


# Converting Gender Column into catagorical 1 and 0 #####

# replace Male with 1 and Female with 0
patients$Gender[patients$Gender == "Female"] <- 0
patients$Gender[patients$Gender == "Male"] <- 1

# converting into numeric variables
patients$Gender <- as.numeric(patients$Gender)

# Show final Gender column
summary(as.factor(patients$Gender))


# Drop Observations with NA in columns ####

patients <- patients %>%
  drop_na()
summary(is.na(patients))


# Density Plots of Variables ####

patients %>%
  select(-c("Gender", "Dataset")) %>%
  pivot_longer(names(.)) %>% 
  ggplot(aes(x = value)) +
  geom_density() +
  facet_wrap(~name, scales = "free")


# Boxplot of Attributes (Before Standardization)

patients %>%
  select(-c("Gender", "Dataset")) %>%
  melt() %>%
  ggplot() +
  geom_boxplot(aes(x = variable, y = value)) +
  labs(
    title = "Boxplot of Attributes (Before Standardization)"
  ) +
  theme(
    axis.text.x = element_text(angle = 90)
  )


# Standardize Patients Data Set
# Create function for standardization
standardize = function(x){
  z <- (x - mean(x)) / sd(x)
  return( z)
}

# Standardize features except gender and dataset, which are factors ####

patients_std <- patients

patients_std[!names(patients_std) %in% c("Gender", "Dataset")] <- apply(
  patients_std[!names(patients_std) %in% c("Gender", "Dataset")], 2, standardize)

patients_std <- as.data.frame(patients_std)  # Convert back into dataframe


# Boxplot of Attributes (After Standardization)
patients_std %>%
  select(-c("Gender", "Dataset")) %>%
  melt() %>%
  ggplot() +
  geom_boxplot(aes(x = variable, y = value)) +
  labs(
    title = "Boxplot of Attributes (After Standardization)"
  ) +
  theme(
    axis.text.x = element_text(angle = 90)
  )


# Splitting Data Set into Testing and Training Set
set.seed(53, sample.kind = "Rounding")

test_index <- createDataPartition(patients_std$Dataset, times = 1, p = 0.1, list = FALSE)

test_att <- patients_std[test_index,] %>% select(-"Dataset")  # Attributes for test set
test_dis <- patients_std[test_index,] %>% select("Dataset")   # Disease indicator for test set

train_att <- patients_std[-test_index,] %>% select(-"Dataset")  # Attributes for train set
train_dis <- patients_std[-test_index,] %>% select("Dataset")   # Disease indicator for train set

# Convert prediction result as factor
test_dis <- as.factor(test_dis$Dataset)
train_dis <- as.factor(train_dis$Dataset)


# Checking proportion of patients in test and train set
tibble(
  "Patients Proportion in Train Set" = mean(as.numeric(train_dis) == 2),
  "Patients Proportion in Test Set" = mean(as.numeric(test_dis) == 2),
)


# Logistic Regression: All features ####

set.seed(63, sample.kind = "Rounding") # if using R 3.6 or later

ctrl <- trainControl(method = "cv", number = 5)  # train control features

train_glm <- train(train_att, train_dis, method = "glm", trControl = ctrl)
glm_preds_f <- predict(train_glm, test_att)

cfm_glm <- confusionMatrix(data = glm_preds_f, reference = test_dis)  # confusion matrix comparing predicted results with actual results

cfm_glm


# Create Table for Storing Full Attribute Results####

if(!exists("result_summary")){
  result_summary <- tibble("Model" = "Logistic Regression (Full attributes)", 
                           "Accuracy" = mean(glm_preds_f == test_dis))
}else{
  result_summary <- rbind(result_summary, 
                          c("Logistic Regression (Full attributes)", 
                            mean(glm_preds_f == test_dis)))
}


# Logistic Regression: Reduced Attributes
set.seed(63, sample.kind = "Rounding") # if using R 3.6 or later

# Remove gender and total_bilirubin columns in train set
train_att_reduced <- train_att %>%
  select(-c("Gender", "Total_Bilirubin"))

# Remove gener and total_bilirubin columns in test set
test_att_reduced <- test_att %>%
  select(-c("Gender", "Total_Bilirubin"))


ctrl <- trainControl(method = "cv", number = 5)

train_glm_1 <- train(train_att_reduced, train_dis, method = "glm", trControl = ctrl)
glm_preds_r <- predict(train_glm_1, test_att)

cfm_glm <- confusionMatrix(data = glm_preds_r, reference = test_dis)


# Add Reduced Attributes Result to Summary Table ####

if(!exists("result_summary")){
  result_summary <- tibble("Model" = "Logistic Regression (Reduced Attributes)", 
                           "Accuracy" = mean(glm_preds_r == test_dis))
}else{
  result_summary <- rbind(result_summary, 
                          c("Logistic Regression (Reduced Attributes)", 
                            mean(glm_preds_r == test_dis)))
}


# KNN Model ########

set.seed(63, sample.kind = "Rounding")

ctrl <- trainControl(method = "cv", number = 5)

train_knn <- train(train_att_reduced, train_dis, method = "knn",
                   trControl = ctrl)

knn_preds <- predict(train_knn, test_att)


# knn confusion Matrix #######

cfm_knn <- confusionMatrix(data = knn_preds, reference = test_dis)


# Add knn Result to Summary Table ####

if(!exists("result_summary")){
  result_summary <- tibble("Model" = "KNN (Reduced Attributes)", 
                           "Accuracy" = mean(knn_preds == test_dis))
}else{
  result_summary <- rbind(result_summary, 
                          c("KNN (Reduced Attributes)", 
                            mean(knn_preds == test_dis)))
}


# Random Forest Model #########

set.seed(63, sample.kind = "Rounding")

train_rf <- train(train_att_reduced, train_dis, method = "rf", ntree = 10)

rf_preds <- predict(train_rf, test_att)

cfm_rf <- confusionMatrix(data = rf_preds, reference = test_dis)


# Add RF Result to Summary Table

if(!exists("result_summary")){
  result_summary <- tibble("Model" = "Random Forest (Reduced Attributes)", 
                           "Accuracy" = mean(rf_preds == test_dis))
}else{
  result_summary <- rbind(result_summary, 
                          c("Random Forest (Reduced Attributes)", 
                            mean(rf_preds == test_dis)))
}


# k-Means ############

#predict function taking in k_means object
predict_kmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}


#k_means model building
set.seed(63, sample.kind = "Rounding")
k <- kmeans(train_att_reduced, centers = 3, nstart = 25)
kmeans_preds <- ifelse(predict_kmeans(test_att_reduced, k) == 1, "1", "0")
cfm_kmeans <- confusionMatrix(data = as.factor(kmeans_preds), reference = test_dis)


# Add k-means Result to Summary Table

if(!exists("result_summary")){
  result_summary <- tibble("Model" = "K-Means (Reduced Attributes)", 
                           "Accuracy" = mean(kmeans_preds == test_dis))
}else{
  result_summary <- rbind(result_summary, 
                          c("K-Means (Reduced Attributes)", 
                            mean(kmeans_preds == test_dis)))
}


# Ensemble ############
en_pred <- ifelse(
  as.numeric(kmeans_preds == 1)+
    as.numeric(glm_preds_r == 1)+
    as.numeric(knn_preds == 1)+
    as.numeric(rf_preds == 1)
  > 2,
  1,0
)

cfm_en <- confusionMatrix(data = as.factor(en_pred), reference = test_dis)


# Add Ensemble Result to Summary Table

if(!exists("result_summary")){
  result_summary <- tibble("Model" = "Ensemble", 
                           "Accuracy" = mean(en_pred == test_dis))
}else{
  result_summary <- rbind(result_summary, 
                          c("Ensemble", 
                            mean(en_pred == test_dis)))
}


# Visualizing Confusion Matrices

layout_matrix <- matrix(c(1:5,0), ncol = 2)  # Create layout matrix with empty Plot 6

layout(layout_matrix)  # Specify Layout

# Plot 1
fourfoldplot(as.table(cfm_glm),
             main = "Logistic Regression Model")
# Plot 2
fourfoldplot(as.table(cfm_knn),
             main = "KNN Model")
# Plot 3
fourfoldplot(as.table(cfm_rf),
             main = "Random Forest Model")
# Plot 4
fourfoldplot(as.table(cfm_kmeans),
             main = "K-Means Model")
# Plot 5
fourfoldplot(as.table(cfm_en),
             main = "Ensemble")