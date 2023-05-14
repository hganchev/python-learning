# problem description:
# Compute - accuracy, precision, recall, f1, rounded to 4 decimal (round(x,4))
# Output - 0.8038, 0.7819, 0.6813, 0.7281
# Sample input - 233,65,109,480

tp = 233    #TRUE POSITIVE
fp = 65     #FALSE POSITIVE
fn = 109    #FALSE NEGATIVE
tn = 480    #TRUE NEGATIVE

total = tp + fp + fn + tn
accuracy = (tp+tn)/total
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*precision*recall/(precision+recall)

print(round(accuracy,4))
print(round(precision,4))
print(round(recall,4))
print(round(f1_score,4))