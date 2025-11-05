from typing import Dict, List


class RealisticCategoryHierarchy:
    """
    Grocery-specific product taxonomy based on real retail operations.
    Categories designed for strategic assortment planning.
    """
    
    @staticmethod
    def create_category_hierarchy() -> Dict:
        """
        Create realistic 3-level grocery category hierarchy.
        Level 1: Department (Fresh, Pantry, etc.)
        Level 2: Category (Dairy, Produce, etc.)
        Level 3: Subcategory (Milk, Yogurt, etc.)
        """
        return {
            'Fresh': {
                'Produce': {
                    'Fruits': ['Apples', 'Bananas', 'Oranges', 'Berries', 'Tropical Fruits', 'Melons', 'Stone Fruits'],
                    'Vegetables': ['Leafy Greens', 'Root Vegetables', 'Tomatoes', 'Peppers', 'Onions', 'Carrots', 'Broccoli'],
                    'Herbs': ['Fresh Herbs', 'Packaged Herbs', 'Specialty Herbs']
                },
                'Dairy': {
                    'Milk': ['Whole Milk', 'Skim Milk', '2% Milk', 'Organic Milk', 'Lactose-Free Milk', 'Plant-Based Milk'],
                    'Yogurt': ['Greek Yogurt', 'Regular Yogurt', 'Flavored Yogurt', 'Probiotic Yogurt', 'Plant-Based Yogurt'],
                    'Cheese': ['Cheddar', 'Mozzarella', 'Parmesan', 'Specialty Cheese', 'Cottage Cheese', 'Cream Cheese'],
                    'Butter': ['Salted Butter', 'Unsalted Butter', 'Organic Butter', 'Vegan Butter'],
                    'Eggs': ['Large Eggs', 'Extra Large Eggs', 'Organic Eggs', 'Free Range Eggs', 'Egg Whites']
                },
                'Meat': {
                    'Poultry': ['Chicken Breast', 'Chicken Thighs', 'Ground Chicken', 'Whole Chicken', 'Turkey'],
                    'Beef': ['Ground Beef', 'Steaks', 'Roasts', 'Beef Ribs', 'Organic Beef'],
                    'Pork': ['Pork Chops', 'Ground Pork', 'Bacon', 'Sausages', 'Ham'],
                    'Seafood': ['Salmon', 'Shrimp', 'Tilapia', 'Tuna', 'Shellfish']
                },
                'Bakery': {
                    'Bread': ['White Bread', 'Wheat Bread', 'Artisan Bread', 'Rolls', 'Baguettes'],
                    'Pastries': ['Croissants', 'Muffins', 'Donuts', 'Danish', 'Cookies'],
                    'Cakes': ['Birthday Cakes', 'Cupcakes', 'Sheet Cakes', 'Specialty Cakes']
                }
            },
            'Pantry': {
                'Canned_Goods': {
                    'Vegetables': ['Canned Corn', 'Canned Beans', 'Canned Tomatoes', 'Canned Peas', 'Canned Carrots'],
                    'Fruits': ['Canned Peaches', 'Canned Pears', 'Fruit Cocktail', 'Canned Pineapple'],
                    'Soups': ['Tomato Soup', 'Chicken Soup', 'Vegetable Soup', 'Beef Soup', 'Specialty Soups'],
                    'Proteins': ['Canned Tuna', 'Canned Salmon', 'Canned Chicken', 'Spam']
                },
                'Dry_Goods': {
                    'Pasta': ['Spaghetti', 'Penne', 'Macaroni', 'Specialty Pasta', 'Whole Wheat Pasta'],
                    'Rice': ['White Rice', 'Brown Rice', 'Basmati Rice', 'Jasmine Rice', 'Wild Rice'],
                    'Cereals': ['Corn Flakes', 'Oatmeal', 'Granola', 'Kids Cereals', 'Health Cereals'],
                    'Flour': ['All-Purpose Flour', 'Wheat Flour', 'Bread Flour', 'Specialty Flours'],
                    'Baking': ['Sugar', 'Baking Powder', 'Baking Soda', 'Yeast', 'Chocolate Chips']
                },
                'Beverages': {
                    'Soft_Drinks': ['Cola', 'Lemon-Lime', 'Orange Soda', 'Diet Sodas', 'Energy Drinks'],
                    'Juices': ['Orange Juice', 'Apple Juice', 'Cranberry Juice', 'Grape Juice', 'Vegetable Juice'],
                    'Water': ['Bottled Water', 'Sparkling Water', 'Flavored Water', 'Premium Water'],
                    'Coffee_Tea': ['Ground Coffee', 'Coffee Beans', 'Tea Bags', 'Specialty Teas', 'Instant Coffee']
                },
                'Snacks': {
                    'Chips': ['Potato Chips', 'Tortilla Chips', 'Corn Chips', 'Pita Chips', 'Veggie Chips'],
                    'Crackers': ['Saltines', 'Graham Crackers', 'Cheese Crackers', 'Wheat Crackers'],
                    'Cookies': ['Chocolate Chip', 'Oatmeal', 'Sandwich Cookies', 'Premium Cookies'],
                    'Candy': ['Chocolate Bars', 'Gummy Candy', 'Hard Candy', 'Mints'],
                    'Nuts': ['Peanuts', 'Almonds', 'Cashews', 'Mixed Nuts', 'Trail Mix']
                },
                'Condiments': {
                    'Sauces': ['Ketchup', 'Mustard', 'Mayonnaise', 'BBQ Sauce', 'Hot Sauce', 'Soy Sauce'],
                    'Oils': ['Olive Oil', 'Vegetable Oil', 'Canola Oil', 'Coconut Oil', 'Specialty Oils'],
                    'Vinegars': ['White Vinegar', 'Apple Cider Vinegar', 'Balsamic Vinegar', 'Rice Vinegar'],
                    'Spices': ['Salt', 'Pepper', 'Garlic Powder', 'Onion Powder', 'Paprika', 'Cumin']
                }
            },
            'Personal_Care': {
                'Health_Beauty': {
                    'Hair_Care': ['Shampoo', 'Conditioner', 'Hair Spray', 'Hair Gel', 'Hair Color'],
                    'Skin_Care': ['Face Wash', 'Moisturizer', 'Sunscreen', 'Anti-Aging', 'Acne Treatment'],
                    'Oral_Care': ['Toothpaste', 'Toothbrushes', 'Mouthwash', 'Dental Floss', 'Whitening'],
                    'Body_Care': ['Body Wash', 'Bar Soap', 'Deodorant', 'Lotion', 'Body Spray'],
                    'Cosmetics': ['Foundation', 'Lipstick', 'Mascara', 'Eye Shadow', 'Nail Polish']
                },
                'Household': {
                    'Cleaning': ['All-Purpose Cleaner', 'Dish Soap', 'Laundry Detergent', 'Bleach', 'Glass Cleaner'],
                    'Paper_Products': ['Toilet Paper', 'Paper Towels', 'Tissues', 'Napkins'],
                    'Trash_Bags': ['Kitchen Bags', 'Tall Bags', 'Lawn Bags', 'Compostable Bags']
                },
                'Baby_Care': {
                    'Diapers': ['Newborn Diapers', 'Size 1', 'Size 2', 'Size 3', 'Size 4', 'Size 5'],
                    'Baby_Food': ['Infant Formula', 'Baby Cereal', 'Purees', 'Snacks'],
                    'Baby_Care': ['Baby Wipes', 'Baby Lotion', 'Baby Shampoo', 'Diaper Cream']
                },
                'Pet_Care': {
                    'Dog_Food': ['Dry Dog Food', 'Wet Dog Food', 'Dog Treats', 'Puppy Food'],
                    'Cat_Food': ['Dry Cat Food', 'Wet Cat Food', 'Cat Treats', 'Kitten Food'],
                    'Pet_Supplies': ['Pet Toys', 'Litter', 'Pet Grooming', 'Pet Accessories']
                }
            },
            'General_Merchandise': {
                'Electronics': {
                    'Accessories': ['Phone Chargers', 'Headphones', 'Cables', 'Screen Protectors', 'Power Banks'],
                    'Small_Electronics': ['Batteries', 'Light Bulbs', 'Extension Cords', 'USB Drives']
                },
                'Home': {
                    'Kitchen': ['Storage Containers', 'Utensils', 'Kitchen Towels', 'Oven Mitts', 'Cutting Boards'],
                    'Storage': ['Plastic Bins', 'Organizers', 'Hooks', 'Shelving'],
                    'Decor': ['Candles', 'Picture Frames', 'Vases', 'Throw Pillows']
                },
                'Seasonal': {
                    'Holiday': ['Christmas Decor', 'Halloween Candy', 'Easter Eggs', 'Fourth of July'],
                    'Outdoor': ['Grilling Supplies', 'Picnic Items', 'Beach Gear', 'Camping Supplies']
                }
            }
        }
    
    @staticmethod
    def create_brand_portfolio() -> Dict[str, List[str]]:
        """Create realistic brand portfolio with market positioning"""
        return {
            'National_Premium': [
                'BrandLeader', 'PremiumChoice', 'QualityFirst', 'BestValue', 'TopTier',
                'EliteSelection', 'PrimeProducts', 'SuperiorGoods', 'MasterBrand', 'UltraQuality'
            ],
            'National_Value': [
                'SmartBuy', 'ValuePack', 'EconoChoice', 'BudgetBest', 'SaveMore',
                'ThriftyPick', 'WiseShopper', 'DealFinder', 'PriceRight', 'CostSaver'
            ],
            'Store_Brand': [
                'StoreSelect', 'HouseChoice', 'OurBrand', 'RetailBest', 'StoreValue',
                'MarketChoice', 'ShopSmart', 'PrivateLabel', 'StoreFavorite', 'OwnBrand'
            ],
            'Organic_Natural': [
                'NaturePure', 'OrganicLife', 'GreenChoice', 'EarthFirst', 'PureNatural',
                'WholeLife', 'CleanEating', 'NaturalWay', 'EcoFriendly', 'BioSelect'
            ],
            'Specialty_Premium': [
                'GourmetSelect', 'ArtisanChoice', 'CraftedGoods', 'SpecialtyPick', 'UniqueFinds',
                'RareTreasures', 'ExoticChoice', 'LuxuryLine', 'DeluxeSelection', 'FinestPick'
            ]
        }
    
    @staticmethod
    def get_all_brands() -> List[str]:
        """Get flattened list of all brands"""
        portfolio = RealisticCategoryHierarchy.create_brand_portfolio()
        return [brand for brands in portfolio.values() for brand in brands]

