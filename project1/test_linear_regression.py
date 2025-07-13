import linear_regression
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Create a linear regression model
    model = linear_regression.LinearRegression()
    
    print("Linear Regression Model with Pybind11")
    print("------------------------------------")
    
    # Get input from user
    try:
        # Input X values
        x_input = input("Enter X values (comma separated): ")
        X = [float(x.strip()) for x in x_input.split(',')]
        
        # Input y values
        y_input = input("Enter y values (comma separated): ")
        y = [float(y.strip()) for y in y_input.split(',')]
        
        # Train the model
        model.fit(X, y)
        
        print("\nModel trained successfully!")
        print(f"Slope (coefficient): {model.get_slope():.4f}")
        print(f"Intercept: {model.get_intercept():.4f}")
        
        # Make predictions
        print("\nMaking predictions:")
        while True:
            try:
                pred_input = input("Enter a value to predict (or 'q' to quit, 'a' for all predictions): ")
                if pred_input.lower() == 'q':
                    break
                elif pred_input.lower() == 'a':
                    predictions = model.predict_vector(X)
                    for x_val, pred in zip(X, predictions):
                        print(f"For x = {x_val:.2f}, predicted y = {pred:.2f}")
                else:
                    x_val = float(pred_input)
                    prediction = model.predict(x_val)
                    print(f"For x = {x_val:.2f}, predicted y = {prediction:.2f}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
        
        # Plotting
        plt.scatter(X, y, color='blue', label='Actual data')
        
        # Generate points for the regression line
        x_min, x_max = min(X), max(X)
        x_line = np.linspace(x_min, x_max, 100)
        y_line = [model.predict(x) for x in x_line]
        
        plt.plot(x_line, y_line, color='red', label='Regression line')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()