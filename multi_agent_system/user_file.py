from pydantic import BaseModel, Field
from typing import List


class UserProfile(BaseModel):
    customer_id: str = Field(
        description="The customer ID of the customer"
    )
    invest_preferences: List[str] = Field(
        description="The invest preferences of the customer"
    )


