from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from smolagents import Tool, CodeAgent, LiteLLMModel, LogLevel

# --- Knowledge base: fine-grained chunks, not one canned answer per category ---
# Each entry is a distinct FACT/ANGLE, not a full solution. The LLM has to combine
# several of these (often across categories) to produce advice for a specific case.
FINANCIAL_KNOWLEDGE = [
    # Mortgage Arrears
    {"topic": "Mortgage Arrears", "kind": "definition", "text": "Mortgage arrears occur when a borrower misses one or more scheduled mortgage payments, causing the outstanding balance owed to the lender to grow beyond the normal loan schedule."},
    {"topic": "Mortgage Arrears", "kind": "regulation", "text": "In the UK, lenders must follow the FCA's Mortgages and Home Finance: Conduct of Business (MCOB) rules, which require them to treat customers in arrears fairly and consider forbearance options before starting repossession action."},
    {"topic": "Mortgage Arrears", "kind": "remedy", "text": "Common lender forbearance options include a temporary payment holiday, extending the mortgage term to lower monthly payments, switching to interest-only for a period, or capitalising the arrears into the loan balance."},
    {"topic": "Mortgage Arrears", "kind": "remedy", "text": "Borrowers can contact a free debt charity such as StepChange or Citizens Advice to get an independent income and expenditure assessment before negotiating with the lender."},
    {"topic": "Mortgage Arrears", "kind": "process", "text": "Before starting court repossession proceedings, UK lenders are required to follow the pre-action protocol, which includes providing clear arrears statements and giving the borrower a genuine opportunity to propose a repayment plan."},
    {"topic": "Mortgage Arrears", "kind": "risk", "text": "Missed mortgage payments are reported to credit reference agencies and can significantly lower a borrower's credit score for up to six years, affecting future borrowing."},
    {"topic": "Mortgage Arrears", "kind": "support_scheme", "text": "The UK Support for Mortgage Interest (SMI) scheme offers an interest-free loan to eligible benefit claimants to help cover mortgage interest payments, repayable when the property is sold."},

    # Credit Card Debt Spiral
    {"topic": "Credit Card Debt Spiral", "kind": "definition", "text": "A credit card debt spiral happens when minimum payments barely cover accruing interest, so the outstanding balance stays flat or grows even though payments are being made."},
    {"topic": "Credit Card Debt Spiral", "kind": "remedy", "text": "Balance transfer cards with a 0% introductory period can pause interest accrual, giving the borrower a window to pay down principal faster."},
    {"topic": "Credit Card Debt Spiral", "kind": "remedy", "text": "The debt avalanche method (paying off the highest-interest debt first) minimises total interest paid, while the debt snowball method (smallest balance first) can improve motivation and adherence."},
    {"topic": "Credit Card Debt Spiral", "kind": "risk", "text": "Persistent minimum-payment behaviour can trigger a 'persistent debt' intervention from UK card issuers, who are required by FCA rules to contact customers who have paid more in interest and fees than principal over 18 months."},

    # Payday Loan Trap
    {"topic": "Payday Loan Trap", "kind": "definition", "text": "A payday loan trap occurs when a borrower repeatedly rolls over or takes out new short-term loans to repay previous ones, with high APRs compounding the total owed."},
    {"topic": "Payday Loan Trap", "kind": "regulation", "text": "UK payday lenders are subject to an FCA price cap limiting total interest and fees to 0.8% per day and total repayment to no more than 100% of the amount borrowed."},
    {"topic": "Payday Loan Trap", "kind": "remedy", "text": "A breathing space scheme can pause interest, fees, and enforcement action for up to 60 days while the borrower gets debt advice and sets up a repayment plan."},

    # Council Tax / Utility Arrears
    {"topic": "Council Tax / Utility Arrears", "kind": "definition", "text": "Council tax or utility arrears arise when household bills for local services or energy go unpaid, often escalating quickly to court summons or disconnection notices."},
    {"topic": "Council Tax / Utility Arrears", "kind": "process", "text": "UK councils typically issue a reminder notice after one missed payment; a second missed payment can result in the loss of the right to pay in instalments, making the full year's tax due immediately."},
    {"topic": "Council Tax / Utility Arrears", "kind": "remedy", "text": "Local welfare assistance schemes and hardship funds run by councils and energy suppliers can provide grants or repayment plans for households in genuine financial hardship."},

    # Overdraft Dependency
    {"topic": "Overdraft Dependency", "kind": "definition", "text": "Overdraft dependency describes a pattern where a person relies on an arranged or unarranged overdraft every month to cover essential spending, with fees and interest reducing available income further."},
    {"topic": "Overdraft Dependency", "kind": "remedy", "text": "Consolidating an overdraft into a lower-interest personal loan can reduce the daily interest cost and create a fixed repayment schedule instead of an open-ended balance."},

    # Student Loan Default
    {"topic": "Student Loan Default", "kind": "definition", "text": "Student loan default occurs when scheduled repayments are missed for an extended period, which can trigger penalty fees, damaged credit, and referral to collections."},
    {"topic": "Student Loan Default", "kind": "remedy", "text": "Many student loan systems offer income-driven repayment plans that recalculate monthly payments based on current income, which can prevent default during periods of low earnings."},

    # Business Cash Flow Crisis
    {"topic": "Business Cash Flow Crisis", "kind": "definition", "text": "A business cash flow crisis happens when a company cannot meet short-term obligations like payroll or supplier invoices despite being profitable on paper, usually due to timing mismatches between receivables and payables."},
    {"topic": "Business Cash Flow Crisis", "kind": "remedy", "text": "Invoice financing or factoring allows a business to receive a large percentage of an unpaid invoice's value immediately from a lender, improving short-term liquidity."},
    {"topic": "Business Cash Flow Crisis", "kind": "remedy", "text": "Renegotiating supplier payment terms to extend the payment window, or offering early-payment discounts to customers, can help close short-term cash flow gaps."},

    # Unexpected Medical Debt
    {"topic": "Unexpected Medical Debt", "kind": "definition", "text": "Unexpected medical debt arises when treatment costs are not fully covered by insurance, leaving the patient responsible for a large and often unplanned balance."},
    {"topic": "Unexpected Medical Debt", "kind": "remedy", "text": "Many hospitals and clinics offer interest-free payment plans or financial hardship programs that can be requested directly from the billing department."},

    # Retirement Savings Shortfall
    {"topic": "Retirement Savings Shortfall", "kind": "definition", "text": "A retirement savings shortfall means an individual's accumulated pension and savings are insufficient to sustain their expected standard of living once they stop working."},
    {"topic": "Retirement Savings Shortfall", "kind": "remedy", "text": "Delaying retirement by even a few years, increasing pension contributions to capture employer matching, and reviewing investment allocation for growth can materially close a projected shortfall."},

    # Identity Theft / Fraudulent Debt
    {"topic": "Identity Theft / Fraudulent Debt", "kind": "definition", "text": "Fraudulent debt occurs when credit or loans are taken out in someone's name without their knowledge or consent, often discovered only when collections activity begins."},
    {"topic": "Identity Theft / Fraudulent Debt", "kind": "remedy", "text": "Victims should report the fraud to the lender and to credit reference agencies immediately, request a copy of the fraudulent credit agreement, and place a notice of dispute or fraud alert on their credit file."},
]


class QdrantQueryTool(Tool):
    name = "qdrant_query"
    description = (
        "Semantic search over a knowledge base of financial-hardship facts, regulations, "
        "remedies, and risks. Each result is a small fragment of knowledge, not a full answer — "
        "you will typically need to run several different queries and combine multiple fragments "
        "to build a complete, well-grounded response."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to the specific angle you want (e.g. definitions, regulations, remedies, risks, support schemes).",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collection_name = "smolagents"
        self.client = QdrantClient(host="127.0.0.1", port=6333, timeout=30)
        self.embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")
        self._seed()

    def _seed(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        vector_size = len(list(self.embedder.embed(["dim probe"]))[0])
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        texts = [d["text"] for d in FINANCIAL_KNOWLEDGE]
        vectors = list(self.embedder.embed(texts))

        points = [
            PointStruct(id=i, vector=vector.tolist(), payload=doc)
            for i, (vector, doc) in enumerate(zip(vectors, FINANCIAL_KNOWLEDGE))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def forward(self, query: str) -> str:
        points = self.client.query_points(
            self.collection_name,
            query=next(self.embedder.query_embed(query)),
            limit=5,
        ).points

        docs = "Retrieved knowledge fragments (sorted by relevance score, highest first):\n" + "".join(
            [
                f"== Fragment {i} (score: {point.score:.4f}) ==\n"
                + f"TOPIC: {point.payload['topic']} | TYPE: {point.payload['kind']}\n"
                + f"TEXT: {point.payload['text']}\n"
                for i, point in enumerate(points)
            ]
        )
        return docs


model = LiteLLMModel(
    model_id="ollama/qwen3-coder:30b",
    api_base="http://localhost:11434",
    num_ctx=8192,
)

agent = CodeAgent(
    tools=[QdrantQueryTool()],
    model=model,
    max_steps=8,
    v,
)

if __name__ == "__main__":
    scenario = (
        "Customer scenario: A homeowner in the UK has missed two consecutive monthly mortgage "
        "payments after a temporary drop in income. They are self-employed, have no savings buffer, "
        "and just received a formal arrears letter from their lender threatening further action if "
        "no payment plan is agreed within 30 days. They also receive some means-tested benefits."
    )

    agent_output = agent.run(
        "You are advising on the following situation:\n\n"
        f"{scenario}\n\n"
        "Use the qdrant_query tool with multiple different queries (e.g. covering definitions, "
        "regulations, remedies, risks, and support schemes relevant to this situation) to gather "
        "relevant knowledge fragments — do not rely on a single query or a single fragment. "
        "Then, based only on the retrieved fragments, write a tailored action plan of 4-6 concrete "
        "steps this specific customer could take, explaining briefly why each step applies to their "
        "circumstances (e.g. their self-employment, lack of savings, benefit eligibility, and the "
        "30-day deadline). Do not just repeat retrieved text verbatim — synthesize it into personalized advice."
    )
    print(agent_output)