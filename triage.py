from laya import Router

# 1. Initialize the router (lazy-loads models on first prediction)
router = Router()

# 2. Define the state (your input data)
state = {
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

# 3. Define your typed questions
questions = {
    # choice: Pick one category from a list
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": [
            "billing: invoices, payments, refunds",
            "technical: bugs, outages, errors",
            "sales: pricing, new plans",
            "other: everything else"
        ]
    },
    # noul: Yes/No with probability
    "wants_refund": {
        "type": "noul",
        "instructions": "Does the customer explicitly ask for a refund?"
    },
    # score: Ordinal rating on a scale
    "urgency": {
        "type": "score",
        "instructions": "How urgent is the request?",
        "criteria": ["routine", "soon", "blocking"]
    }
}

# 4. Run inference (one forward pass for all questions)
result = router.predict(state, questions)

# 5. Inspect the structured output
print("Department:", result["answers"]["department"]["choice"])
print("  Confidence:", result["answers"]["department"]["confidence"])
print("Refund Requested:", result["answers"]["wants_refund"]["noul"])
print("Urgency Score:", result["answers"]["urgency"]["score"])
print("Routing Model Used:", result["routing"]["model"])