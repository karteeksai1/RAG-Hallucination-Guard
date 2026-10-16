from laya import Router

# 1. Initialize the router. It lazily loads checkpoints on first predict.
# For production, consider Router(preload=True) to avoid cold-start delays.
router = Router()

def triage_ticket(subject: str, body: str):
    """Triage a single support ticket and return structured decisions."""
    
    state = {
        "subject": subject,
        "body": body,
    }

    # 2. Define typed questions: choice, score, and noul (yes/no).
    questions = {
        # choice: pick exactly one label from a set of options
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this support ticket?",
            "criteria": [
                "billing: charges, invoices, payments, refunds, cancelling a paid plan",
                "technical: bugs, errors, outages, features not working, login issues",
                "account: password reset, two-factor authentication, profile settings",
                "other: sales questions, discounts, feedback, everything else",
            ],
        },
        # score: ordinal rating on a defined scale
        "urgency": {
            "type": "score",
            "instructions": "How urgent is the request?",
            "criteria": [
                "routine: no time pressure, can wait days",
                "soon: needs attention within 24 hours",
                "blocking: customer cannot work right now, needs immediate fix",
            ],
        },
        # noul: calibrated yes/no probability
        "wants_refund": {
            "type": "noul",
            "instructions": "Does the customer explicitly ask for a refund or money back?",
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the customer threaten to cancel or leave?",
        },
    }

    # 3. Run all questions in ONE forward pass (~13-40 ms on GPU).
    result = router.predict(state, questions)
    answers = result["answers"]

    # 4. Extract structured outputs.
    dept = answers["department"]["choice"]
    dept_conf = answers["department"]["confidence"]
    urgency = answers["urgency"]["score"]
    urgency_conf = answers["urgency"]["confidence"]
    refund_p = answers["wants_refund"]["noul"]
    churn_p = answers["churn_risk"]["noul"]

    return {
        "department": dept,
        "department_confidence": dept_conf,
        "urgency": urgency,
        "urgency_confidence": urgency_conf,
        "wants_refund": refund_p,
        "churn_risk": churn_p,
        "routing_model": result["routing"]["model"],  # e.g., "english" or "multilingual"
    }


def main():
    # Example tickets (replace with your own data)
    tickets = [
        {
            "subject": "Duplicate charge on invoice #4411",
            "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
        },
        {
            "subject": "Dashboard down since 9am",
            "body": "The dashboard has been showing a 500 error since this morning. Nobody on our team can log in.",
        },
        {
            "subject": "Question about student discount",
            "body": "Do you offer a discount for students? I'm interested in the pro plan.",
        },
        {
            "subject": "Password reset not working",
            "body": "I requested a password reset but never received the email. Can you help?",
        },
    ]

    print("=" * 70)
    print("SUPPORT TICKET TRIAGE RESULTS")
    print("=" * 70)

    for i, ticket in enumerate(tickets, 1):
        result = triage_ticket(ticket["subject"], ticket["body"])

        print(f"\n--- Ticket {i}: {ticket['subject']} ---")
        print(f"  Department : {result['department']} (conf: {result['department_confidence']:.2f})")
        print(f"  Urgency    : {result['urgency']} (conf: {result['urgency_confidence']:.2f})")
        print(f"  Refund?    : {result['wants_refund']:.3f} (P(true))")
        print(f"  Churn risk : {result['churn_risk']:.3f} (P(true))")
        print(f"  Routed via : {result['routing_model']}")

        # 5. Simple confidence-gated routing logic
        if result["department_confidence"] >= 0.85:
            print(f"  ➜ Action   : Auto-route to {result['department'].upper()}")
        else:
            print(f"  ➜ Action   : ESCALATE (low dept confidence)")

        if result["wants_refund"] > 0.7:
            print(f"  ➜ FLAG     : Refund requested (p={result['wants_refund']:.2f})")

        if result["churn_risk"] > 0.7:
            print(f"  ➜ FLAG     : Churn risk high (p={result['churn_risk']:.2f})")


if __name__ == "__main__":
    main()