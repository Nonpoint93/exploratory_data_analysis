from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AirbnbModel(Base):
    __tablename__ = 'airbnb'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    host_id = Column(Integer)
    host_name = Column(String)
    neighbourhood_group = Column(String)
    neighbourhood = Column(String)
    latitude = Column(Integer)
    longitude = Column(Integer)
    room_type = Column(String)
    price = Column(Integer)
    minimum_nights = Column(Integer)
    number_of_reviews = Column(Integer)
    last_review = Column(String)
    reviews_per_month = Column(Integer)
    calculated_host_listings_count = Column(Integer)
    availability_365 = Column(Integer)