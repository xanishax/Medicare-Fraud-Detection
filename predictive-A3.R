# load libraries
library(C50)
library(class)
library(caret)
library(data.table)
library(dplyr)
library(e1071)
library(GGally)
library(ggplot2)
library(kernlab)
library(lubridate)
library(mlr3)
library(mlr3learners)
library(mlr3measures)
library(nnet) # for SL.nnet
library(randomForest)
library(SuperLearner)
library(tidyverse)


Train_Beni <- read.csv("Train-Ben.csv")
Train_Inpatient <- read.csv("Train-In.csv")
Train_Outpatient <- read.csv("Train-Out.csv")
Train_results <- read.csv("Train-Results.csv")

Test_Beni <- read.csv("Test-Ben.csv")
Test_Inpatient <- read.csv("Test-In.csv")
Test_Outpatient <- read.csv("Test-Out.csv")
Test_results <- read.csv("Test-Results.csv")


current_date <- Sys.Date()  # Get current date
Train_Beni$Age <- round(as.numeric(interval(Train_Beni$DOB, current_date) / dyears(1)),0)

# Convert Date of Birth column to Date object
Train_Beni$DOB <- as.Date(Train_Beni$DOB)

# Create Patient_Age_Year column
Train_Beni$Patient_Age_Year <- year(Train_Beni$DOB)

# Create Patient_Age_Month column
Train_Beni$Patient_Age_Month <- month(Train_Beni$DOB)

Train_Beni$Race <- as.character(Train_Beni$Race)

Train_Beni$RenalDiseaseIndicator <- ifelse(Train_Beni$RenalDiseaseIndicator == 'Y', 1,
                                           ifelse(Train_Beni$RenalDiseaseIndicator == '0', 2, Train_Beni$RenalDiseaseIndicator))

Train_Beni$ChronicCond_KidneyDisease <- as.character(Train_Beni$ChronicCond_KidneyDisease)

Train_Beni$State <- as.character(Train_Beni$State)

# Calculate the counts for each state
state_counts <- table(Train_Beni$State)

# Sort the state counts in descending order
sorted_states <- names(sort(state_counts, decreasing = TRUE))

# Reorder the levels of the State variable based on the sorted states
Train_Beni$State <- factor(Train_Beni$State, levels = sorted_states)

Train_Beni$ChronicCond_Alzheimer <- as.character(Train_Beni$ChronicCond_Alzheimer)

Train_Beni$ChronicCond_Heartfailure <- as.character(Train_Beni$ChronicCond_Heartfailure)



#OUTPATIENT DATA

# Convert ClaimEndDt and ClaimStartDt to Date objects
Train_Outpatient$ClaimEndDt <- as.Date(Train_Outpatient$ClaimEndDt)
Train_Outpatient$ClaimStartDt <- as.Date(Train_Outpatient$ClaimStartDt)

# Calculate Claim_Duration in days
Train_Outpatient$Claim_Duration <- as.numeric(difftime(Train_Outpatient$ClaimEndDt, Train_Outpatient$ClaimStartDt, units = "days"))

#INPATIENT
Train_Inpatient$ClaimEndDt <- as.Date(Train_Inpatient$ClaimEndDt)
Train_Inpatient$ClaimStartDt <- as.Date(Train_Inpatient$ClaimStartDt)
Train_Inpatient$Claim_Duration <- as.numeric(difftime(Train_Inpatient$ClaimEndDt, Train_Inpatient$ClaimStartDt, units = "days"))


Train_Inpatient$AdmissionDt <- as.Date(Train_Inpatient$AdmissionDt)
Train_Inpatient$DischargeDt <- as.Date(Train_Inpatient$DischargeDt)
Train_Inpatient$Admission_Duration <- as.numeric(difftime(Train_Inpatient$DischargeDt, Train_Inpatient$AdmissionDt, units = "days"))
Train_Inpatient$Admission_Duration <- ifelse(is.na(Train_Inpatient$Admission_Duration), 0, Train_Inpatient$Admission_Duration)

Train_Inpatient$PatientType <- 1
Train_Outpatient$PatientType <- 0
Train_Beni$Is_deceased <- ifelse(is.na(Train_Beni$DOD), 0, 1)


merged_data1 <- merge(Train_Beni, Train_Outpatient, by = "BeneID", all.x = TRUE)
merged_data2 <- merge(Train_Beni, Train_Inpatient, by = "BeneID", all.x = TRUE)
final_merged_data <- bind_rows(merged_data1, merged_data2)
final <- merge(final_merged_data, Train_results, by = "Provider", all.x = TRUE)


# Remove rows with missing values in the Potential_Fraud column
final_filtered <- final[!is.na(final$PotentialFraud), ]
final_filtered$Admission_Duration <- ifelse(is.na(final_filtered$Admission_Duration), 0, final_filtered$Admission_Duration)

# Create a new column 'Address' by combining 'State' and 'County' with a space
final_filtered$Address <- paste(final_filtered$State, final_filtered$County, sep = " ")

# Remove 'State' and 'County' columns from the dataframe
final_filtered <- subset(final_filtered, select = -c(State, County))

# Define the list of features you want to keep
selected_features <- c("Race", "Address", "Age", "Is_deceased", "PatientType", "ChronicCond_KidneyDisease", 
                       "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_IschemicHeart", 
                       "ChronicCond_stroke", "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt", 
                       "OPAnnualDeductibleAmt",  "InscClaimAmtReimbursed", "DeductibleAmtPaid",
                       "OperatingPhysician", "OtherPhysician", 
                       "ClmAdmitDiagnosisCode", "Claim_Duration",  "DiagnosisGroupCode", 
                       "Admission_Duration", "PotentialFraud")

# Create a new dataset with only the selected features
featuresex <- subset(final_filtered, select = selected_features)



# List of variables to be frequency encoded
variables_to_encode <- c("Address")

# Apply frequency encoding
for (var in variables_to_encode) {
  freq <- table(featuresex[[var]])
  featuresex[[paste0(var, "_freq")]] <- freq[as.character(featuresex[[var]])]
}


# Remove Address
featuresex <- subset(featuresex, select = -Address)
new_order <- c("Race", "Address_freq", "Age", "Is_deceased", "PatientType", "ChronicCond_KidneyDisease", 
               "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_IschemicHeart", 
               "ChronicCond_stroke", "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt", 
               "OPAnnualDeductibleAmt",  "InscClaimAmtReimbursed", "DeductibleAmtPaid",
               "OperatingPhysician", "OtherPhysician", 
               "ClmAdmitDiagnosisCode", "Claim_Duration",  "DiagnosisGroupCode", 
               "Admission_Duration", "PotentialFraud")

featuresex <- featuresex[, new_order]


# Convert "Yes" to 1 and "No" to 0 in the PotentialFraud column
featuresex$PotentialFraud <- ifelse(featuresex$PotentialFraud == "Yes", 1, 0)


# Target Encoding

# Convert data frame to a data table
setDT(featuresex)





# Calculate mean of 'PotentialFraud' for each category of 'OperatingPhysician'
target_means <- featuresex[, .(mean_PotentialFraud = mean(PotentialFraud)), by = OperatingPhysician]

# Merge target means with original data
featuresex <- merge(featuresex, target_means, by = "OperatingPhysician", all.x = TRUE)

# Replace 'AttendingPhysician' with the calculated mean
featuresex$OperatingPhysician <- featuresex$mean_PotentialFraud

# Remove the redundant column
featuresex$mean_PotentialFraud <- NULL



# Calculate mean of 'PotentialFraud' for each category of 'OtherPhysician'
target_means <- featuresex[, .(mean_PotentialFraud = mean(PotentialFraud)), by = OtherPhysician]

# Merge target means with original data
featuresex <- merge(featuresex, target_means, by = "OtherPhysician", all.x = TRUE)

# Replace 'OtherPhysician' with the calculated mean
featuresex$OtherPhysician <- featuresex$mean_PotentialFraud

# Remove the redundant column
featuresex$mean_PotentialFraud <- NULL





# Calculate mean of 'PotentialFraud' for each category of 'ClmAdmitDiagnosisCode'
target_means <- featuresex[, .(mean_PotentialFraud = mean(PotentialFraud)), by = ClmAdmitDiagnosisCode]

# Merge target means with original data
featuresex <- merge(featuresex, target_means, by = "ClmAdmitDiagnosisCode", all.x = TRUE)

# Replace 'ClmAdmitDiagnosisCode' with the calculated mean
featuresex$ClmAdmitDiagnosisCode <- featuresex$mean_PotentialFraud

# Remove the redundant column
featuresex$mean_PotentialFraud <- NULL



# Calculate mean of 'PotentialFraud' for each category of 'DiagnosisGroupCode'
target_means <- featuresex[, .(mean_PotentialFraud = mean(PotentialFraud)), by = DiagnosisGroupCode]

# Merge target means with original data
featuresex <- merge(featuresex, target_means, by = "DiagnosisGroupCode", all.x = TRUE)

# Replace 'DiagnosisGroupCode' with the calculated mean
featuresex$DiagnosisGroupCode <- featuresex$mean_PotentialFraud

# Remove the redundant column
featuresex$mean_PotentialFraud <- NULL




# Factoring
featuresex$Race <- as.factor(featuresex$Race)
featuresex$Address_freq <- as.numeric(featuresex$Address_freq)
featuresex$ChronicCond_KidneyDisease <- as.factor(featuresex$ChronicCond_KidneyDisease)
featuresex$ChronicCond_Cancer <- as.factor(featuresex$ChronicCond_Cancer)
featuresex$ChronicCond_ObstrPulmonary <- as.factor(featuresex$ChronicCond_ObstrPulmonary)
featuresex$ChronicCond_IschemicHeart <- as.factor(featuresex$ChronicCond_IschemicHeart)
featuresex$ChronicCond_stroke <- as.factor(featuresex$ChronicCond_stroke)
featuresex$IPAnnualReimbursementAmt <- as.numeric(featuresex$IPAnnualReimbursementAmt)
featuresex$IPAnnualDeductibleAmt <- as.numeric(featuresex$IPAnnualDeductibleAmt)
featuresex$OPAnnualDeductibleAmt <- as.numeric(featuresex$OPAnnualDeductibleAmt)
featuresex$Age <- as.numeric(featuresex$Age)
featuresex$Is_deceased <- as.factor(featuresex$Is_deceased)
featuresex$InscClaimAmtReimbursed <- as.numeric(featuresex$InscClaimAmtReimbursed)

featuresex$OperatingPhysician <- as.numeric(featuresex$OperatingPhysician)
featuresex$OtherPhysician <- as.numeric(featuresex$OtherPhysician)

featuresex$DiagnosisGroupCode <- as.numeric(featuresex$DiagnosisGroupCode)
featuresex$DeductibleAmtPaid <- as.numeric(featuresex$DeductibleAmtPaid)
featuresex$ClmAdmitDiagnosisCode <- as.numeric(featuresex$ClmAdmitDiagnosisCode)

featuresex$PatientType <- as.factor(featuresex$PatientType)
featuresex$Claim_Duration <- as.numeric(featuresex$Claim_Duration)
featuresex$Admission_Duration <- as.numeric(featuresex$Admission_Duration)
featuresex$PotentialFraud <- as.factor(featuresex$PotentialFraud)

featuresex <- featuresex[complete.cases(featuresex)]



# Create the scatterplot with separate colors for PotentialFraud
ggplot(featuresex, aes(x = OperatingPhysician, y = OtherPhysician, color = as.factor(PotentialFraud))) +
  geom_point(alpha = 0.5) +
  labs(title = "Scatterplot of OperatingPhysician and OtherPhysician",
       x = "Operating Physician",
       y = "Other Physician",
       color = "Potential Fraud") +
  theme_minimal() +
  scale_color_manual(values = c("blue", "red"), labels = c("No Fraud", "Fraud"))



# Fit a random forest model
rf_model <- randomForest(PotentialFraud ~ ., data = sampled_data, importance = TRUE)

# Get the importance of features
importance(rf_model)

# Plot the importance of features
varImpPlot(rf_model)


# Set seed for reproducibility
set.seed(123)

# Randomly select 100,000 rows
sampled_data <- featuresex[sample(nrow(featuresex), 100000), ]

# Check the dimensions of the sampled data
dim(sampled_data)



# Scaling
sampled_data$Address_freq <- scale(sampled_data$Address_freq, center = TRUE, scale = TRUE)
sampled_data$IPAnnualReimbursementAmt <- scale(sampled_data$IPAnnualReimbursementAmt, center = TRUE, scale = TRUE)
sampled_data$IPAnnualDeductibleAmt <- scale(sampled_data$IPAnnualDeductibleAmt, center = TRUE, scale = TRUE)
sampled_data$OPAnnualDeductibleAmt <- scale(sampled_data$OPAnnualDeductibleAmt, center = TRUE, scale = TRUE)
sampled_data$Age <- scale(sampled_data$Age, center = TRUE, scale = TRUE)
sampled_data$InscClaimAmtReimbursed <- scale(sampled_data$InscClaimAmtReimbursed, center = TRUE, scale = TRUE)
sampled_data$OperatingPhysician <- scale(sampled_data$OperatingPhysician, center = TRUE, scale = TRUE)
sampled_data$OtherPhysician <- scale(sampled_data$OtherPhysician, center = TRUE, scale = TRUE)
sampled_data$DiagnosisGroupCode <- scale(sampled_data$DiagnosisGroupCode, center = TRUE, scale = TRUE)
sampled_data$DeductibleAmtPaid <- scale(sampled_data$DeductibleAmtPaid, center = TRUE, scale = TRUE)
sampled_data$ClmAdmitDiagnosisCode <- scale(sampled_data$ClmAdmitDiagnosisCode, center = TRUE, scale = TRUE)
sampled_data$Claim_Duration <- scale(sampled_data$Claim_Duration, center = TRUE, scale = TRUE)
sampled_data$Admission_Duration <- scale(sampled_data$Admission_Duration, center = TRUE, scale = TRUE)


train.size <- .7
train.indices <- sample(x = seq(1, nrow(sampled_data), by = 1), size =
                          ceiling(train.size * nrow(sampled_data)), replace = FALSE)
trainData <- sampled_data[ train.indices, ]
testData <- sampled_data[ -train.indices, ]


# Fit a Naive Bayes classifier
fraud.learner.nb <- lrn("classif.naive_bayes")

# Create a Naive Bayes classifier
fraud.learner.nb <- naiveBayes(PotentialFraud ~ ., data = trainData, laplace = 1)

# Make predictions on test data
fraud.pred.nb <- predict(fraud.learner.nb, newdata = testData)

# Compute accuracy
accuracy <- sum(fraud.pred.nb == testData$PotentialFraud) / length(testData$PotentialFraud)

# Create a confusion matrix
conf_matrix <- confusionMatrix(table(predicted = fraud.pred.nb, actual = testData$PotentialFraud))
F_score <- conf_matrix$byClass["F1"]

# Print accuracy and confusion matrix
print(paste("Accuracy:", accuracy))
print("Confusion Matrix:")
print(conf_matrix)




# CLASSIFIER 2

# Ensure PotentialFraud is a factor
featuresex$PotentialFraud <- as.factor(featuresex$PotentialFraud)

# Plot the data
ggplot(data = featuresex, aes(x = OperatingPhysician, y = OtherPhysician, color = PotentialFraud, shape = PotentialFraud)) + 
  geom_point(size = 2) +
  scale_color_manual(values=c("#000000", "#FF0000")) +
  theme_minimal() +
  labs(title = "Scatterplot of OperatingPhysician and OtherPhysician",
       x = "Operating Physician",
       y = "Other Physician")


# Take a smaller subset of trainData for training
set.seed(123)  # For reproducibility
subset_indices <- sample(1:nrow(trainData), size = floor(0.1 * nrow(trainData)))  # Using 10% of the data
train_subset <- trainData[subset_indices, ]


set.seed(1)
tune.out = tune(svm, PotentialFraud~. , data = train_subset , kernel = "polynomial", 
                ranges = list(cost = c(0.1, 1, 10, 100, 1000)))
summary(tune.out)

# Fit SVM with polynomial kernel
svm.model <- svm(PotentialFraud ~ ., data = trainData, kernel = "polynomial",  cost = 1)
svm.pred = predict(svm.model,testData[, -22])

svm.results = confusionMatrix(table(predicted = svm.pred,
                                    actual = testData$PotentialFraud))
svm.results
F_score <- svm.results$byClass["F1"]


# CLASSIFIER 3

fraud.task <- TaskClassif$new(id = "fraud", backend = trainData,
                              target = "PotentialFraud")
# run experiment
k.values <- rev(c(50, 55, 60, 65, 70, 75, 80, 85))
# k.values <- rev(c(30, 35, 40, 45, 50, 55, 60))
storage <- data.frame(matrix(NA, ncol = 3, nrow = length(k.values)))
colnames(storage) <- c("acc_train", "acc_test", "k")
for (i in 1:length(k.values)) {
  fraud.learner <- lrn("classif.kknn", k = k.values[i])
  fraud.learner$train(task = fraud.task)
  # test data
  # choose additional adequate measures from: mlr3::mlr_measures
  fraud.pred <- fraud.learner$predict_newdata(newdata = testData)
  storage[i, "acc_test"] <- fraud.pred$score(msr("classif.acc"))
  # train data
  fraud.pred <- fraud.learner$predict_newdata(newdata = trainData)
  storage[i, "acc_train"] <- fraud.pred$score(msr("classif.acc"))
  storage[i, "k"] <- k.values[i]
}


storage <- storage[rev(order(storage$k)), ]
plot(
  x = storage$k, y = storage$acc_train, main = "Overfitting behavior
KNN",
  xlab = "k - the number of neighbors to consider", ylab = "accuracy",
  col = "blue", type = "l",
  xlim = rev(range(storage$k)),
  ylim = c(
    min(storage$acc_train, storage$acc_test),
    max(storage$acc_train, storage$acc_test)
  ),
  log = "x"
)
lines(x = storage$k, y = storage$acc_test, col = "orange")
legend("topleft", c("test", "train"), col = c("orange", "blue"), lty = 1)


# Fit KNN with K=60
fraud.learner.knn <- lrn("classif.kknn", k = 60)
fraud.learner.knn$train(task = fraud.task)
fraud.pred.knn <- fraud.learner.knn$predict_newdata(newdata = testData)
knn.results = confusionMatrix(table(predicted = fraud.pred.knn$response,
                                    actual = testData$PotentialFraud))

knn.results
F_score <- knn.results$byClass["F1"]

# CLASSIFIER 4

# Train a Random Forest model
rf_model <- randomForest(PotentialFraud ~ ., data = trainData, ntree = 100)

# Make predictions on the test data
predictions <- predict(rf_model, newdata = testData)

# Compute accuracy
accuracy <- sum(predictions == testData$PotentialFraud) / length(testData$PotentialFraud)

# Create a confusion matrix
conf_matrix <- confusionMatrix(table(predicted = predictions, actual = testData$PotentialFraud))

# Print accuracy
print(paste("Accuracy:", accuracy))

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

F_score <- conf_matrix$byClass["F1"]



# FEATURE IMPORTANCE

# Extract feature importances from the Random Forest model
importance <- rf_model$importance

# Sort the feature importances in descending order
importance <- importance[order(importance, decreasing = TRUE), ]

# Print the sorted feature importances
print(importance)

# Create the barplot without x-axis labels
barplot_heights <- barplot(
  importance, 
  main = "Feature Importances", 
  las = 2,         # Rotate labels to be perpendicular to the axis
  names.arg = NA,  # No x-axis labels
  ylim = c(0, max(importance) * 1.2) # Add some space for labels
)

# Add rotated labels manually
labels <- names(importance)
text(x = barplot_heights, y = -0.02 * max(importance), labels = labels, srt = 45, adj = 1, xpd = TRUE, cex = 0.8)


# Convert the feature importances to a dataframe
importance_df <- data.frame(Feature = rownames(importance), Importance = importance)
