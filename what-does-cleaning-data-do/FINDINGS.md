# Findings

Data: the Titanic passenger list, Kaggle training set, 891 passengers (mirror:
datasciencedojo/datasets on GitHub; a public historical record compiled from
Encyclopedia Titanica). Model: logistic regression on class, sex, age,
family aboard, log fare and port (plus a cabin flag and the title where
stated), scored by 10 x 5-fold cross-validation; fill values are learned on
each training fold only.

## The blanks
866 blank cells: Age 177, Cabin 687, Embarked 2. Overall survival 38.4%.

## Delete every passenger with a blank
183 of 891 remain: 158 first class, 15 second, 10 third. Survival reads
**67.2%**. Cabin blanks are mostly third class (cabin on record: 1st 176/216,
2nd 16/184, 3rd 12/491), so deleting blanks deletes the poorest passengers.
A model trained on those 183 scores **73.5%** on the other 708.

## A blank can be a clue
Cabin on record: 66.7% survived. Cabin blank: 30.0%.

## Ages
Dropping, mean-filling or title-filling the 177 blank ages gives about the
same accuracy (79.4-79.5%, sd ~2.3).

## Text hides data
Titles (Mr / Mrs / Miss / Master) as columns, plus the cabin flag: **83.0%**
(from 79.4%).

## Zero is not blank; big is not wrong
15 passengers paid £0 (whether that is a real price, the record does not
say). The highest fare, £512.33, belongs to 3 passengers; all survived.

## Scoreboard (on passengers the model did not learn from)
Guess everyone died 61.6% · train only on the 183 complete rows 73.5% ·
keep everyone and read the data 83.0%.
