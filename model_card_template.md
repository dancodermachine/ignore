# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model estimates whether a person's yearly income is above or below $50,000, based on basic demographic and employment information. It was created by Dan in November 2026, and this is version 1.0.

The model is a random forest classifier. It combines the votes of 100 decision trees, each of which learns simple rules such as "people with a graduate degree who work more than 45 hours a week tend to earn more." Before training, text answers such as occupation or marital status are converted into numbers the model can read. The data preparation step and the model are packaged together and saved as a single file, `model/model.pkl`.

The model is used through a web API. A user sends one person's details and receives an answer of either `>50K` or `<=50K`.

## Intended Use

The model is intended for learning and demonstration. It shows how a machine learning model can be trained, tested, and deployed as a web service. It is suitable for exploring income patterns in historical census data and for teaching MLOps practices.

The model must not be used for any decision that affects real people, such as hiring, lending, insurance, housing, or eligibility for services. The data is decades old, and the model reflects historical inequalities, as described under Ethical Considerations.

## Training Data

The model was trained on the UCI "Adult" Census Income dataset, which comes from the 1994 US Census database. The dataset contains 32,561 people, about 24% of whom earn more than $50K.

The model uses 14 pieces of information about each person: age, type of employer, education level, years of education, marital status, occupation, household relationship, race, sex, capital gains, capital losses, hours worked per week, country of origin, and a census sampling weight.

A random 80% of the records (26,048 people) was used for training. The split was made so that both parts keep the same share of high earners.

## Evaluation Data

The remaining 20% of the records (6,513 people) was held back and never shown to the model during training. This data was used only to measure how well the model performs on people it has not seen before.

## Metrics

The model was evaluated with four metrics on the held-back data. Accuracy is the share of people the model classifies correctly. Precision is the share of people predicted to earn over $50K who really do. Recall is the share of real high earners that the model finds. The F1 score combines precision and recall into a single number, where higher is better.

| Metric | Score | In plain terms |
|---|---|---|
| Accuracy | 85.7% | The model is right for about 6 in 7 people. |
| Precision | 73.7% | When the model says someone earns over $50K, it is right about 3 times in 4. |
| Recall | 63.1% | Of the people who really earn over $50K, the model finds about 2 in 3. |
| F1 score | 68.0% | Precision and recall together are moderately good. |

Accuracy alone makes the model look better than it is. Because only about a quarter of people earn over $50K, a model that always answered "under $50K" would already be right 76% of the time. Precision, recall, and F1 give a fairer picture, and they show that the model misses about a third of high earners.

The model was also checked separately for each value of every categorical feature, and the full results are in `slice_output.txt`. The table below shows the results by sex and race.

| Group | People | Precision | Recall | F1 |
|---|---|---|---|---|
| Male | 4,355 | 73.3% | 63.8% | 68.2% |
| Female | 2,158 | 76.0% | 59.6% | 66.8% |
| White | 5,533 | 73.6% | 63.7% | 68.3% |
| Black | 662 | 77.6% | 55.9% | 65.0% |
| Asian-Pacific Islander | 200 | 68.8% | 63.5% | 66.0% |
| American Indian / Eskimo | 73 | 66.7% | 44.4% | 53.3% |

The model misses high earners more often among women, Black people, and American Indian / Eskimo people than among men and white people. The results for small groups are based on few people and should be treated as rough estimates.

## Ethical Considerations

The model uses race, sex, and country of origin as inputs. Using these characteristics in real decisions about people could be unlawful as well as unfair.

The data reflects the pay gaps of 1994. The model learns these patterns and repeats them rather than correcting them. As the group results above show, the model is also less reliable for some groups than for others.

The census data is anonymized and publicly available. The API does not store any information that users send to it.

## Caveats and Recommendations

Incomes and prices have changed a great deal since 1994, so the $50K threshold no longer means what it did. The model should be retrained on recent data before any real-world use.

The data comes mostly from US residents, so results for people in other countries are unlikely to be reliable. In the original data, some answers are recorded as "?", and the model treats "?" as an answer of its own rather than filling in the gap.

If the model is retrained or used more widely, its performance should be checked for each group every time, not just overall. Possible improvements include removing sensitive inputs, tuning the model's settings, and adjusting the decision threshold to find more high earners, while comparing fairness and accuracy after each change.
