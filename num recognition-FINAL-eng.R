################################## KNN ################################

library(imager)
library(caret)

image_folder <- "D://nums"

process_image <- function(image_path) {
  img <- load.image(image_path)            
  img_gray <- grayscale(img)              
  img_resized <- resize(img_gray, 28, 28)  
  img_vector <- as.numeric(img_resized)    
  return(img_vector)
}

test_image <- "D://nums//0_1.jpg.JPG"

if (file.exists(test_image)) {
  test_vector <- process_image(test_image)
  print("Image loaded successfully.")
} else {
  print("File not found. Please check the path.")
}

image_files <- list.files(path = image_folder, pattern = "\\.JPG$", full.names = TRUE)

image_data <- data.frame()

for (file in image_files) {
  img_vector <- process_image(file)
  image_data <- rbind(image_data, img_vector)
}

print(head(image_data))

image_data <- data.frame()

for (file in image_files) {
  img_vector <- process_image(file)
  image_data <- rbind(image_data, img_vector)
}

labels <- sapply(image_files, function(x) as.numeric(sub("_.+", "", basename(x))))
image_data$label <- as.factor(labels)

print(head(image_data))

library(tidymodels)

set.seed(123)
Split8020 <- initial_split(image_data, prop = 0.8, strata = label)
DataTrain <- training(Split8020)
DataTest <- testing(Split8020)

RecipeImage <- recipe(label ~ ., data = DataTrain) |>
  step_normalize(all_predictors()) |> 
  step_naomit()       

ModelDesignKNN <- nearest_neighbor(neighbors = 1, weight_func = "rectangular") |>
  set_engine("kknn") |>           
  set_mode("classification")           

library(kknn)

WFModelImage <- workflow() |>
  add_recipe(RecipeImage) |>
  add_model(ModelDesignKNN) |>
  fit(DataTrain)

DataTestWithPred <- augment(WFModelImage, DataTest)
head(DataTestWithPred)

ConfMatrixImage <- conf_mat(DataTestWithPred, truth = label, estimate = .pred_class)
print(ConfMatrixImage)

accuracy <- sum(diag(ConfMatrixImage$table)) / sum(ConfMatrixImage$table)
cat("Model accuracy:", round(accuracy * 100, 2), "%\n")


new_test_image <- "D://test//8.jpg"

if (file.exists(new_test_image)) {
  new_test_vector <- process_image(new_test_image)
  print("New handwritten image processed successfully.")
} else {
  stop("New handwritten file not found. Please check the path.")
}

new_test_df <- as.data.frame(t(new_test_vector))  
colnames(new_test_df) <- colnames(DataTrain)[-ncol(DataTrain)]  

new_test_pred <- predict(WFModelImage, new_data = new_test_df)

cat("Predicted digit for the new handwritten image:", new_test_pred$.pred_class, "\n")

############################### MLP (nnet) #####################################

library(imager)
library(caret)
library(tidymodels)
library(nnet)

image_folder <- "D://nums"

process_image <- function(image_path) {
  img <- load.image(image_path)           
  img_gray <- grayscale(img)              
  img_resized <- resize(img_gray, 28, 28) 
  img_vector <- as.numeric(img_resized)    
  return(img_vector)
}

image_files <- list.files(path = image_folder, pattern = "\\.JPG$", full.names = TRUE)

image_data <- data.frame()

for (file in image_files) {
  img_vector <- process_image(file)
  image_data <- rbind(image_data, img_vector)
}

labels <- sapply(image_files, function(x) as.numeric(sub("_.+", "", basename(x))))
image_data$label <- as.factor(labels)

set.seed(123)
Split8020 <- initial_split(image_data, prop = 0.8, strata = label)
DataTrain <- training(Split8020)
DataTest <- testing(Split8020)

library(tidymodels)
library(recipes)

RecipeImagePCA <- recipe(label ~ ., data = DataTrain) |>
  step_normalize(all_predictors()) |>  
  step_pca(all_predictors(), threshold = 0.95)  

ModelDesignMLP <- mlp(hidden_units = c(6), 
                      epochs = 1000, 
                      penalty = 1) |>
  set_engine("nnet") |>
  set_mode("classification")

set.seed(123)
WFModelImageMLP <- workflow() |>
  add_recipe(RecipeImagePCA) |>
  add_model(ModelDesignMLP) |>
  fit(DataTrain)


DataTestWithPredMLP <- augment(WFModelImageMLP, DataTest)

ConfMatrixImageMLP <- conf_mat(DataTestWithPredMLP, truth = label, estimate = .pred_class)
print(ConfMatrixImageMLP)

accuracy_MLP <- sum(diag(ConfMatrixImageMLP$table)) / sum(ConfMatrixImageMLP$table)
cat("MLP model accuracy:", round(accuracy_MLP * 100, 2), "%\n")

new_test_image <- "D://test//8.jpg"

if (file.exists(new_test_image)) {
  new_test_vector <- process_image(new_test_image)
  print("New handwritten image processed successfully.")
} else {
  stop("New handwritten file not found. Please check the path.")
}

new_test_df <- as.data.frame(t(new_test_vector))  
colnames(new_test_df) <- colnames(DataTrain)[-ncol(DataTrain)]  

new_test_pred_MLP <- predict(WFModelImageMLP, new_data = new_test_df)

cat("Predicted digit for the new handwritten image:", new_test_pred_MLP$.pred_class, "\n")

###################################### Multi-Layer Neural Network (Keras) ###################################

library(imager)
library(caret)
library(tidymodels)
library(keras)
library(tensorflow)

image_folder <- "D://nums"

process_image <- function(image_path) {
  img <- load.image(image_path)           
  img_gray <- grayscale(img)               
  img_resized <- resize(img_gray, 28, 28) 
  img_vector <- as.numeric(img_resized)    
  return(img_vector)
}

image_files <- list.files(path = image_folder, pattern = "\\.JPG$", full.names = TRUE)

image_data <- data.frame()

for (file in image_files) {
  img_vector <- process_image(file)
  image_data <- rbind(image_data, img_vector)
}

labels <- sapply(image_files, function(x) as.numeric(sub("_.+", "", basename(x))))
image_data$label <- as.factor(labels)

set.seed(123)
Split8020 <- initial_split(image_data, prop = 0.8, strata = label)
DataTrain <- training(Split8020)
DataTest <- testing(Split8020)

x_train <- as.matrix(DataTrain[,-ncol(DataTrain)])  
y_train <- model.matrix(~ DataTrain$label - 1)

x_test <- as.matrix(DataTest[,-ncol(DataTest)])
y_test <- model.matrix(~ DataTest$label - 1)


set_random_seed(123)
model <- keras_model_sequential()

model$add(layer_dense(units = 200, activation = 'relu', input_shape = c(ncol(DataTrain) - 1))) 
model$add(layer_dense(units = 155, activation = 'relu'))
model$add(layer_dense(units = length(unique(DataTrain$label)), activation = 'softmax')) 

model$compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = list("accuracy")
)


library(reticulate)
np <- import("numpy")

x_train <- np$array(x_train, dtype = "float32")
x_test  <- np$array(x_test, dtype = "float32")
y_train <- np$array(y_train, dtype = "float32")
y_test  <- np$array(y_test, dtype = "float32")

history <- model$fit(
  x = x_train,
  y = y_train,
  epochs = 180L,       
  batch_size = 32L,    
  validation_data = list(x_test, y_test)
)

score <- model$evaluate(x_test, y_test)
cat("Model accuracy:", score[[2]] * 100, "%\n")


new_image_path <- "D://test//8.jpg"   

new_img_vector <- process_image(new_image_path)

new_img_vector <- matrix(new_img_vector, nrow = 1)  

prediction <- model$predict(new_img_vector)

predicted_class <- which.max(prediction) - 1  

cat("Predicted class:", predicted_class, "\n")

##################################### CNN ###################################

library(imager)
library(caret)
library(tidymodels)
library(keras)
library(tensorflow)

image_folder <- "D://nums"

process_image <- function(image_path) {
  img <- load.image(image_path)            
  img_gray <- grayscale(img)               
  img_resized <- resize(img_gray, 28, 28)  
  img_vector <- as.numeric(img_resized)    
  return(img_vector)
}

image_files <- list.files(path = image_folder, pattern = "\\.JPG$", full.names = TRUE)

image_data <- data.frame()

for (file in image_files) {
  img_vector <- process_image(file)
  image_data <- rbind(image_data, img_vector)
}

labels <- sapply(image_files, function(x) as.numeric(sub("_.+", "", basename(x))))
image_data$label <- as.factor(labels)

set.seed(123)
Split8020 <- initial_split(image_data, prop = 0.8, strata = label)
DataTrain <- training(Split8020)
DataTest <- testing(Split8020)

x_train <- as.matrix(DataTrain[,-ncol(DataTrain)])  
x_train <- array(x_train, dim = c(nrow(x_train), 28, 28, 1))
y_train <- model.matrix(~ DataTrain$label - 1)

x_test <- as.matrix(DataTest[,-ncol(DataTest)])
x_test <- array(x_test, dim = c(nrow(x_test), 28, 28, 1))
y_test <- model.matrix(~ DataTest$label - 1)

set_random_seed(123)
model_cnn <- keras_model_sequential()

model_cnn$add(layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)))
model_cnn$add(layer_max_pooling_2d(pool_size = c(2, 2)))
model_cnn$add(layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu"))
model_cnn$add(layer_max_pooling_2d(pool_size = c(2, 2)))
model_cnn$add(layer_flatten())
model_cnn$add(layer_dense(units = 128, activation = "relu"))
model_cnn$add(layer_dense(units = length(unique(DataTrain$label)), activation = "softmax"))


model_cnn$compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = list("accuracy")
)

library(reticulate)
np <- import("numpy")

x_train <- np$array(x_train, dtype = "float32")
x_test  <- np$array(x_test, dtype = "float32")
y_train <- np$array(y_train, dtype = "float32")
y_test  <- np$array(y_test, dtype = "float32")

history <- model_cnn$fit(
  x = x_train,
  y = y_train,
  epochs = 200L,       
  batch_size = 32L,   
  validation_data = list(x_test, y_test)
)

score <- model_cnn$evaluate(x_test, y_test)
cat("Model accuracy:", score[[2]] * 100, "%\n")

new_image_path <- "D://test//8.jpg"  

new_img_vector <- process_image(new_image_path)

new_img_vector <- array(new_img_vector, dim = c(1, 28, 28, 1))

prediction <- model_cnn$predict(new_img_vector)

predicted_class <- which.max(prediction) - 1  

cat("Predicted class:", predicted_class, "\n")