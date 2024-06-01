import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

X=np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]).reshape(-1,1)
y=np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])

def probcalc(model,X):
    log_odds=model.coef_*X+model.intercept_
    odds=np.exp(log_odds)
    probability=odds/(1+odds)
    return probability

model=linear_model.LogisticRegression()
model.fit(X,y)

print("Check whether Ankush can lift a given weight or not")

a=int(input("Enter weight in kgs."))

predicted=model.predict(np.array([a]).reshape(-1,1))

if predicted[0]==1:
    print("Ankush can lift the weight.")
else:
    print("Ankush cannot life the weight.")

log_odds=model.coef_
odds=np.exp(log_odds)
print("The odds are: \n",odds)

print("The probability of each weight that can be lifted by Ankush is : \n",probcalc(model,X))
probability=probcalc(model,np.array([a]).reshape(-1,1))

plt.scatter(X, y, color='blue', label='Data Points')

X_test = np.linspace(0, 110, 300).reshape(-1, 1)
y_prob = probcalc(model, X_test)
plt.plot(X_test, y_prob, color='red', label='Logistic Regression Curve')
plt.scatter([a], [probability], color='green',label=f'Input Weight {a} kg')
plt.xlabel('Weight (kgs)')
plt.ylabel('Probability of lifting')
plt.title('Logistic Regression for Weight Lifting')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
