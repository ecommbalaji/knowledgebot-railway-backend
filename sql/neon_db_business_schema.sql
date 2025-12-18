-- Neon DB Business Database Schema
-- This mimics a business database with customer, product, order, and analytics data
-- Run this SQL file in your Neon DB instance

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Customers table (business customer data - NO PII, only anonymized IDs)
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_tier VARCHAR(50), -- bronze, silver, gold, platinum
    registration_date DATE,
    total_orders INTEGER DEFAULT 0,
    lifetime_value DECIMAL(10, 2) DEFAULT 0.00,
    preferred_category VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Products table (product catalog)
CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    price DECIMAL(10, 2) NOT NULL,
    cost DECIMAL(10, 2),
    stock_quantity INTEGER DEFAULT 0,
    description TEXT,
    brand VARCHAR(100),
    rating DECIMAL(3, 2), -- 0.00 to 5.00
    review_count INTEGER DEFAULT 0,
    is_available BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Orders table (order transactions)
CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    order_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    order_status VARCHAR(50), -- pending, processing, shipped, delivered, cancelled
    total_amount DECIMAL(10, 2) NOT NULL,
    shipping_cost DECIMAL(10, 2) DEFAULT 0.00,
    discount_amount DECIMAL(10, 2) DEFAULT 0.00,
    payment_method VARCHAR(50),
    shipping_address_country VARCHAR(100),
    shipping_address_region VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Order items table (line items in orders)
CREATE TABLE IF NOT EXISTS order_items (
    order_item_id VARCHAR(50) PRIMARY KEY,
    order_id VARCHAR(50) REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id VARCHAR(50) REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    total_price DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Sales analytics table (aggregated sales data)
CREATE TABLE IF NOT EXISTS sales_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    category VARCHAR(100),
    total_revenue DECIMAL(12, 2) DEFAULT 0.00,
    total_orders INTEGER DEFAULT 0,
    total_units_sold INTEGER DEFAULT 0,
    average_order_value DECIMAL(10, 2),
    top_product_id VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, category)
);

-- Inventory table (warehouse inventory)
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id VARCHAR(50) PRIMARY KEY,
    product_id VARCHAR(50) REFERENCES products(product_id),
    warehouse_location VARCHAR(100),
    quantity_available INTEGER DEFAULT 0,
    quantity_reserved INTEGER DEFAULT 0,
    reorder_level INTEGER DEFAULT 10,
    last_restocked_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_order_date ON orders(order_date DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(order_status);
CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_is_available ON products(is_available);
CREATE INDEX IF NOT EXISTS idx_sales_analytics_date ON sales_analytics(date DESC);
CREATE INDEX IF NOT EXISTS idx_inventory_product_id ON inventory(product_id);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_customers_updated_at BEFORE UPDATE ON customers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_inventory_updated_at BEFORE UPDATE ON inventory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- DUMMY DATA INSERTS
-- ============================================

-- Insert dummy customers
INSERT INTO customers (customer_id, customer_tier, registration_date, total_orders, lifetime_value, preferred_category, is_active) VALUES
('CUST001', 'gold', '2023-01-15', 45, 12500.00, 'Electronics', TRUE),
('CUST002', 'silver', '2023-03-22', 23, 5600.00, 'Clothing', TRUE),
('CUST003', 'platinum', '2022-11-10', 78, 23400.00, 'Electronics', TRUE),
('CUST004', 'bronze', '2024-02-05', 8, 1200.00, 'Home & Garden', TRUE),
('CUST005', 'gold', '2023-07-18', 52, 14800.00, 'Books', TRUE),
('CUST006', 'silver', '2023-09-30', 31, 7200.00, 'Clothing', TRUE),
('CUST007', 'bronze', '2024-01-12', 12, 1800.00, 'Electronics', TRUE),
('CUST008', 'platinum', '2022-08-25', 95, 31200.00, 'Electronics', TRUE),
('CUST009', 'gold', '2023-05-14', 38, 9800.00, 'Home & Garden', TRUE),
('CUST010', 'silver', '2023-12-01', 19, 4200.00, 'Books', TRUE)
ON CONFLICT (customer_id) DO NOTHING;

-- Insert dummy products
INSERT INTO products (product_id, product_name, category, subcategory, price, cost, stock_quantity, description, brand, rating, review_count, is_available) VALUES
('PROD001', 'Wireless Bluetooth Headphones', 'Electronics', 'Audio', 129.99, 65.00, 150, 'Premium noise-cancelling headphones with 30-hour battery', 'TechBrand', 4.5, 234, TRUE),
('PROD002', 'Cotton T-Shirt', 'Clothing', 'Tops', 24.99, 8.00, 500, '100% organic cotton, comfortable fit', 'EcoWear', 4.2, 89, TRUE),
('PROD003', 'Smartphone 128GB', 'Electronics', 'Mobile', 699.99, 420.00, 75, 'Latest model with advanced camera system', 'TechBrand', 4.7, 567, TRUE),
('PROD004', 'Garden Tool Set', 'Home & Garden', 'Tools', 49.99, 20.00, 200, 'Complete set of 8 essential garden tools', 'GardenPro', 4.3, 156, TRUE),
('PROD005', 'Mystery Novel - Hardcover', 'Books', 'Fiction', 18.99, 6.00, 300, 'Bestselling mystery thriller', 'BookHouse', 4.6, 412, TRUE),
('PROD006', 'Running Shoes', 'Clothing', 'Footwear', 89.99, 35.00, 120, 'Lightweight running shoes with cushioned sole', 'SportMax', 4.4, 278, TRUE),
('PROD007', 'Laptop Stand', 'Electronics', 'Accessories', 34.99, 12.00, 250, 'Adjustable aluminum laptop stand', 'TechBrand', 4.1, 134, TRUE),
('PROD008', 'Coffee Maker', 'Home & Garden', 'Appliances', 79.99, 40.00, 80, 'Programmable 12-cup coffee maker', 'HomeTech', 4.5, 201, TRUE),
('PROD009', 'Science Fiction Book', 'Books', 'Fiction', 15.99, 5.00, 400, 'Award-winning sci-fi novel', 'BookHouse', 4.8, 523, TRUE),
('PROD010', 'Winter Jacket', 'Clothing', 'Outerwear', 149.99, 60.00, 90, 'Waterproof winter jacket with insulation', 'OutdoorGear', 4.6, 189, TRUE)
ON CONFLICT (product_id) DO NOTHING;

-- Insert dummy orders
INSERT INTO orders (order_id, customer_id, order_date, order_status, total_amount, shipping_cost, discount_amount, payment_method, shipping_address_country, shipping_address_region) VALUES
('ORD001', 'CUST001', '2024-12-01 10:30:00', 'delivered', 129.99, 5.99, 0.00, 'credit_card', 'USA', 'California'),
('ORD002', 'CUST002', '2024-12-02 14:20:00', 'shipped', 74.97, 4.99, 10.00, 'paypal', 'USA', 'New York'),
('ORD003', 'CUST003', '2024-12-03 09:15:00', 'processing', 699.99, 0.00, 50.00, 'credit_card', 'USA', 'Texas'),
('ORD004', 'CUST001', '2024-12-04 16:45:00', 'delivered', 49.99, 5.99, 0.00, 'credit_card', 'USA', 'California'),
('ORD005', 'CUST005', '2024-12-05 11:30:00', 'delivered', 18.99, 3.99, 0.00, 'paypal', 'USA', 'Florida'),
('ORD006', 'CUST006', '2024-12-06 13:20:00', 'shipped', 89.99, 5.99, 0.00, 'credit_card', 'USA', 'Illinois'),
('ORD007', 'CUST003', '2024-12-07 10:00:00', 'processing', 34.99, 4.99, 0.00, 'credit_card', 'USA', 'Texas'),
('ORD008', 'CUST008', '2024-12-08 15:30:00', 'delivered', 229.98, 0.00, 20.00, 'credit_card', 'USA', 'Washington'),
('ORD009', 'CUST009', '2024-12-09 12:15:00', 'shipped', 79.99, 5.99, 0.00, 'paypal', 'USA', 'Oregon'),
('ORD010', 'CUST010', '2024-12-10 09:45:00', 'pending', 15.99, 3.99, 0.00, 'credit_card', 'USA', 'Arizona')
ON CONFLICT (order_id) DO NOTHING;

-- Insert dummy order items
INSERT INTO order_items (order_item_id, order_id, product_id, quantity, unit_price, total_price) VALUES
('OI001', 'ORD001', 'PROD001', 1, 129.99, 129.99),
('OI002', 'ORD002', 'PROD002', 3, 24.99, 74.97),
('OI003', 'ORD003', 'PROD003', 1, 699.99, 699.99),
('OI004', 'ORD004', 'PROD004', 1, 49.99, 49.99),
('OI005', 'ORD005', 'PROD005', 1, 18.99, 18.99),
('OI006', 'ORD006', 'PROD006', 1, 89.99, 89.99),
('OI007', 'ORD007', 'PROD007', 1, 34.99, 34.99),
('OI008', 'ORD008', 'PROD001', 1, 129.99, 129.99),
('OI009', 'ORD008', 'PROD007', 1, 34.99, 34.99),
('OI010', 'ORD008', 'PROD003', 1, 699.99, 699.99),
('OI011', 'ORD009', 'PROD008', 1, 79.99, 79.99),
('OI012', 'ORD010', 'PROD009', 1, 15.99, 15.99)
ON CONFLICT (order_item_id) DO NOTHING;

-- Insert dummy sales analytics
INSERT INTO sales_analytics (date, category, total_revenue, total_orders, total_units_sold, average_order_value, top_product_id) VALUES
('2024-12-01', 'Electronics', 129.99, 1, 1, 129.99, 'PROD001'),
('2024-12-02', 'Clothing', 74.97, 1, 3, 74.97, 'PROD002'),
('2024-12-03', 'Electronics', 699.99, 1, 1, 699.99, 'PROD003'),
('2024-12-04', 'Home & Garden', 49.99, 1, 1, 49.99, 'PROD004'),
('2024-12-05', 'Books', 18.99, 1, 1, 18.99, 'PROD005'),
('2024-12-06', 'Clothing', 89.99, 1, 1, 89.99, 'PROD006'),
('2024-12-07', 'Electronics', 34.99, 1, 1, 34.99, 'PROD007'),
('2024-12-08', 'Electronics', 864.97, 1, 3, 864.97, 'PROD003'),
('2024-12-09', 'Home & Garden', 79.99, 1, 1, 79.99, 'PROD008'),
('2024-12-10', 'Books', 15.99, 1, 1, 15.99, 'PROD009')
ON CONFLICT (date, category) DO NOTHING;

-- Insert dummy inventory
INSERT INTO inventory (inventory_id, product_id, warehouse_location, quantity_available, quantity_reserved, reorder_level, last_restocked_at) VALUES
('INV001', 'PROD001', 'Warehouse A', 150, 0, 20, '2024-11-15 10:00:00'),
('INV002', 'PROD002', 'Warehouse B', 500, 0, 50, '2024-11-20 14:30:00'),
('INV003', 'PROD003', 'Warehouse A', 75, 0, 10, '2024-11-25 09:00:00'),
('INV004', 'PROD004', 'Warehouse C', 200, 0, 25, '2024-11-18 11:00:00'),
('INV005', 'PROD005', 'Warehouse B', 300, 0, 30, '2024-11-22 16:00:00'),
('INV006', 'PROD006', 'Warehouse A', 120, 0, 15, '2024-11-28 10:30:00'),
('INV007', 'PROD007', 'Warehouse C', 250, 0, 30, '2024-11-19 13:00:00'),
('INV008', 'PROD008', 'Warehouse B', 80, 0, 10, '2024-11-24 15:00:00'),
('INV009', 'PROD009', 'Warehouse A', 400, 0, 40, '2024-11-21 12:00:00'),
('INV010', 'PROD010', 'Warehouse C', 90, 0, 12, '2024-11-26 14:00:00')
ON CONFLICT (inventory_id) DO NOTHING;

