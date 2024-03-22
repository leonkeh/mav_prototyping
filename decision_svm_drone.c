#include <stdio.h>
#include <stdlib.h>

#define NUM_FEATURES 136
#define NUM_CLASSES 2

int main() {
    double coefficients[NUM_CLASSES][NUM_FEATURES];
    double intercepts[NUM_CLASSES];
    FILE *file = fopen("decision_svm_parameters.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    // Read coefficients
    for (int i = 0; i < NUM_CLASSES; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (fscanf(file, "%lf", &coefficients[i][j]) != 1) {
                fprintf(stderr, "Error reading coefficients\n");
                return 1;
            }
        }
    }

    // Read intercepts
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (fscanf(file, "%lf", &intercepts[i]) != 1) {
            fprintf(stderr, "Error reading intercepts\n");
            return 1;
        }
    }

    // Implement decision function
    double input[NUM_FEATURES] = {0.5, 0.5}; // Example input
    double decision_value = 0.0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        double dot_product = 0.0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            dot_product += coefficients[i][j] * input[j];
        }
        decision_value += dot_product + intercepts[i];
    }

    // Make prediction based on decision value
    int predicted_class = decision_value > 0 ? 1 : 0;
    printf("Predicted class: %d\n", predicted_class);

    fclose(file);
    return 0;
}