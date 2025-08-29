"""API Data Processor for Component Vendor Interfaces - Week 2 Implementation."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re


class APIProcessor:
    """Processes API data from component vendors (pricing, availability, specifications)."""
    
    def __init__(self):
        self.logger = logging.getLogger('APIProcessor')
        
        # Vendor-specific processing rules
        self.vendor_rules = {
            'digikey': {
                'price_field': 'unit_price',
                'quantity_breaks': True,
                'lead_time_field': 'lead_time_days',
                'currency': 'USD'
            },
            'mouser': {
                'price_field': 'price_usd',
                'quantity_breaks': True,
                'lead_time_field': 'availability',
                'currency': 'USD'
            },
            'farnell': {
                'price_field': 'price',
                'quantity_breaks': True,
                'lead_time_field': 'lead_time',
                'currency': 'GBP'
            },
            'rs': {
                'price_field': 'unit_price',
                'quantity_breaks': False,
                'lead_time_field': 'delivery_time',
                'currency': 'GBP'
            }
        }
        
        self.logger.info("APIProcessor initialized with 4 vendor-specific processing rules")
    
    def process_api_data(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process API data and extract procurement-relevant information."""
        try:
            processed = {}
            
            # Extract vendor information
            vendor_info = self._extract_vendor_info(api_data)
            processed['vendor_info'] = vendor_info
            
            # Process pricing data
            pricing = self._process_pricing(api_data, vendor_info)
            processed['pricing'] = pricing
            
            # Process availability and lead time
            availability = self._process_availability(api_data, vendor_info)
            processed['availability'] = availability
            
            # Process specifications
            specifications = self._process_specifications(api_data)
            processed['specifications'] = specifications
            
            # Generate procurement recommendations
            procurement = self._generate_procurement_recommendations(pricing, availability)
            processed['procurement_recommendations'] = procurement
            
            # Calculate total cost of ownership
            tco = self._calculate_total_cost_ownership(pricing, availability, specifications)
            processed['total_cost_ownership'] = tco
            
            return processed
            
        except Exception as e:
            self.logger.error(f"API processing failed: {e}")
            return {
                'error': str(e),
                'processing_status': 'failed',
                'vendor_info': {'vendor': 'unknown'},
                'pricing': {'price_usd': 0.0, 'price_valid': False},
                'availability': {'stock_quantity': 0, 'lead_time_days': 999}
            }
    
    def _extract_vendor_info(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize vendor information."""
        vendor_info = {
            'vendor': api_data.get('manufacturer', 'Unknown'),
            'part_number': api_data.get('part_number', ''),
            'manufacturer_part_number': api_data.get('manufacturer_part_number', api_data.get('part_number', '')),
            'datasheet_url': api_data.get('datasheet_url', ''),
            'vendor_part_number': api_data.get('vendor_part_number', '')
        }
        
        # Determine vendor type from various fields
        vendor_name = vendor_info['vendor'].lower()
        if 'digikey' in vendor_name or 'digi-key' in vendor_name:
            vendor_info['vendor_type'] = 'digikey'
        elif 'mouser' in vendor_name:
            vendor_info['vendor_type'] = 'mouser'  
        elif 'farnell' in vendor_name:
            vendor_info['vendor_type'] = 'farnell'
        elif 'rs' in vendor_name or 'rs components' in vendor_name:
            vendor_info['vendor_type'] = 'rs'
        else:
            vendor_info['vendor_type'] = 'generic'
        
        return vendor_info
    
    def _process_pricing(self, api_data: Dict[str, Any], vendor_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process pricing information with quantity breaks and currency conversion."""
        pricing = {}
        
        vendor_type = vendor_info.get('vendor_type', 'generic')
        rules = self.vendor_rules.get(vendor_type, self.vendor_rules['digikey'])
        
        # Extract base price
        price_field = rules['price_field']
        unit_price = api_data.get(price_field, api_data.get('price_usd', 0.0))
        
        pricing['unit_price'] = float(unit_price)
        pricing['currency'] = rules['currency']
        
        # Convert to USD if needed
        if rules['currency'] != 'USD':
            pricing['price_usd'] = self._convert_to_usd(unit_price, rules['currency'])
        else:
            pricing['price_usd'] = float(unit_price)
        
        # Process quantity breaks if available
        if rules['quantity_breaks'] and 'quantity_breaks' in api_data:
            breaks = []
            for break_info in api_data['quantity_breaks']:
                breaks.append({
                    'quantity': int(break_info.get('quantity', 1)),
                    'price': float(break_info.get('price', unit_price)),
                    'price_usd': self._convert_to_usd(
                        float(break_info.get('price', unit_price)), 
                        rules['currency']
                    ) if rules['currency'] != 'USD' else float(break_info.get('price', unit_price))
                })
            pricing['quantity_breaks'] = sorted(breaks, key=lambda x: x['quantity'])
        else:
            pricing['quantity_breaks'] = [{'quantity': 1, 'price': unit_price, 'price_usd': pricing['price_usd']}]
        
        # Calculate price analysis
        if len(pricing['quantity_breaks']) > 1:
            pricing['volume_discount_available'] = True
            max_discount = (pricing['quantity_breaks'][0]['price_usd'] - 
                          pricing['quantity_breaks'][-1]['price_usd']) / pricing['quantity_breaks'][0]['price_usd']
            pricing['max_volume_discount_percent'] = max_discount * 100
        else:
            pricing['volume_discount_available'] = False
            pricing['max_volume_discount_percent'] = 0.0
        
        # Price validation
        pricing['price_valid'] = pricing['price_usd'] > 0.0
        pricing['last_updated'] = api_data.get('last_updated', datetime.now().isoformat())
        
        return pricing
    
    def _process_availability(self, api_data: Dict[str, Any], vendor_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process availability and lead time information."""
        availability = {}
        
        vendor_type = vendor_info.get('vendor_type', 'generic')
        rules = self.vendor_rules.get(vendor_type, self.vendor_rules['digikey'])
        
        # Extract stock quantity
        availability['stock_quantity'] = int(api_data.get('stock_quantity', 0))
        availability['minimum_order_quantity'] = int(api_data.get('minimum_order_quantity', 1))
        
        # Extract lead time
        lead_time_field = rules['lead_time_field']
        lead_time_raw = api_data.get(lead_time_field, api_data.get('lead_time_days', 0))
        
        # Normalize lead time to days
        if isinstance(lead_time_raw, str):
            availability['lead_time_days'] = self._parse_lead_time_string(lead_time_raw)
        else:
            availability['lead_time_days'] = int(lead_time_raw)
        
        # Stock status classification
        stock_qty = availability['stock_quantity']
        lead_time = availability['lead_time_days']
        
        if stock_qty > 1000 and lead_time <= 1:
            availability['stock_status'] = 'excellent'
        elif stock_qty > 100 and lead_time <= 7:
            availability['stock_status'] = 'good'
        elif stock_qty > 10 and lead_time <= 14:
            availability['stock_status'] = 'fair'
        elif stock_qty > 0:
            availability['stock_status'] = 'limited'
        else:
            availability['stock_status'] = 'out_of_stock'
        
        # Lead time classification
        if lead_time <= 1:
            availability['lead_time_category'] = 'immediate'
        elif lead_time <= 7:
            availability['lead_time_category'] = 'short'
        elif lead_time <= 30:
            availability['lead_time_category'] = 'medium'
        else:
            availability['lead_time_category'] = 'long'
        
        return availability
    
    def _process_specifications(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process technical specifications from API data."""
        specs = {}
        
        # Common specification fields
        spec_fields = [
            'tolerance', 'temperature_coefficient', 'voltage_rating',
            'current_rating', 'power_rating', 'frequency_range',
            'operating_temperature_min', 'operating_temperature_max',
            'package_type', 'mounting_type'
        ]
        
        for field in spec_fields:
            if field in api_data:
                specs[field] = api_data[field]
        
        # Process temperature range if min/max available
        temp_min = api_data.get('operating_temperature_min')
        temp_max = api_data.get('operating_temperature_max')
        if temp_min is not None and temp_max is not None:
            specs['operating_temp_range'] = {
                'min': float(temp_min),
                'max': float(temp_max),
                'range_c': float(temp_max) - float(temp_min)
            }
        
        # Specification completeness score
        total_possible_specs = len(spec_fields) + 1  # +1 for temp range
        specs_provided = len([v for v in specs.values() if v is not None])
        specs['completeness_score'] = specs_provided / total_possible_specs
        
        return specs
    
    def _generate_procurement_recommendations(self, pricing: Dict[str, Any], availability: Dict[str, Any]) -> Dict[str, Any]:
        """Generate procurement recommendations based on pricing and availability."""
        recommendations = {}
        
        # Procurement priority scoring
        score = 0
        factors = []
        
        # Price factor (lower is better)
        price_usd = pricing.get('price_usd', 999)
        if price_usd < 0.01:
            score += 20
            factors.append('Very low cost')
        elif price_usd < 0.1:
            score += 15
            factors.append('Low cost')
        elif price_usd < 1.0:
            score += 10
            factors.append('Moderate cost')
        else:
            score += 0
            factors.append('Higher cost')
        
        # Availability factor
        stock_status = availability.get('stock_status', 'out_of_stock')
        if stock_status == 'excellent':
            score += 25
            factors.append('Excellent availability')
        elif stock_status == 'good':
            score += 20
            factors.append('Good availability')
        elif stock_status == 'fair':
            score += 10
            factors.append('Fair availability')
        else:
            score += 0
            factors.append('Poor availability')
        
        # Lead time factor
        lead_time_category = availability.get('lead_time_category', 'long')
        if lead_time_category == 'immediate':
            score += 15
            factors.append('Immediate delivery')
        elif lead_time_category == 'short':
            score += 10
            factors.append('Short lead time')
        elif lead_time_category == 'medium':
            score += 5
            factors.append('Medium lead time')
        else:
            score += 0
            factors.append('Long lead time')
        
        # Volume discount factor
        if pricing.get('volume_discount_available', False):
            score += 10
            factors.append('Volume discounts available')
        
        # Final recommendation
        if score >= 50:
            recommendations['recommendation'] = 'strongly_recommended'
        elif score >= 35:
            recommendations['recommendation'] = 'recommended'
        elif score >= 20:
            recommendations['recommendation'] = 'acceptable'
        else:
            recommendations['recommendation'] = 'not_recommended'
        
        recommendations.update({
            'procurement_score': score,
            'factors': factors,
            'suggested_order_quantity': self._suggest_order_quantity(pricing, availability)
        })
        
        return recommendations
    
    def _calculate_total_cost_ownership(self, pricing: Dict[str, Any], availability: Dict[str, Any], 
                                       specifications: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate total cost of ownership including logistics and risk factors."""
        tco = {}
        
        unit_price = pricing.get('price_usd', 0.0)
        lead_time_days = availability.get('lead_time_days', 0)
        
        # Base cost
        tco['unit_cost_usd'] = unit_price
        
        # Logistics cost (estimated based on lead time)
        if lead_time_days <= 7:
            logistics_multiplier = 1.05  # 5% for fast delivery
        elif lead_time_days <= 30:
            logistics_multiplier = 1.02  # 2% for standard delivery
        else:
            logistics_multiplier = 1.01  # 1% for slow delivery
        
        tco['logistics_cost_usd'] = unit_price * (logistics_multiplier - 1)
        
        # Inventory holding cost (based on lead time)
        holding_cost_rate = 0.002  # 0.2% per day
        tco['holding_cost_usd'] = unit_price * holding_cost_rate * lead_time_days
        
        # Risk cost (based on availability)
        stock_status = availability.get('stock_status', 'out_of_stock')
        if stock_status == 'out_of_stock':
            risk_multiplier = 1.20  # 20% risk premium
        elif stock_status == 'limited':
            risk_multiplier = 1.10  # 10% risk premium
        else:
            risk_multiplier = 1.00  # No risk premium
        
        tco['risk_cost_usd'] = unit_price * (risk_multiplier - 1)
        
        # Total TCO
        tco['total_cost_ownership_usd'] = (
            tco['unit_cost_usd'] + 
            tco['logistics_cost_usd'] + 
            tco['holding_cost_usd'] + 
            tco['risk_cost_usd']
        )
        
        tco['tco_premium_percent'] = ((tco['total_cost_ownership_usd'] / unit_price) - 1) * 100
        
        return tco
    
    def _convert_to_usd(self, amount: float, currency: str) -> float:
        """Convert currency to USD (simplified conversion rates)."""
        # Simplified conversion rates - in production, use real-time rates
        conversion_rates = {
            'GBP': 1.25,
            'EUR': 1.08,
            'JPY': 0.0067,
            'CAD': 0.74
        }
        
        rate = conversion_rates.get(currency, 1.0)
        return amount * rate
    
    def _parse_lead_time_string(self, lead_time_str: str) -> int:
        """Parse lead time string to days."""
        lead_time_str = lead_time_str.lower()
        
        # Extract number
        numbers = re.findall(r'\d+', lead_time_str)
        if not numbers:
            return 14  # Default 2 weeks
        
        value = int(numbers[0])
        
        # Convert to days based on unit
        if 'week' in lead_time_str:
            return value * 7
        elif 'month' in lead_time_str:
            return value * 30
        elif 'day' in lead_time_str or 'd' in lead_time_str:
            return value
        else:
            return value  # Assume days if no unit specified
    
    def _suggest_order_quantity(self, pricing: Dict[str, Any], availability: Dict[str, Any]) -> int:
        """Suggest optimal order quantity based on pricing breaks and availability."""
        moq = availability.get('minimum_order_quantity', 1)
        stock_qty = availability.get('stock_quantity', 0)
        
        # Find optimal quantity break
        quantity_breaks = pricing.get('quantity_breaks', [])
        if len(quantity_breaks) > 1:
            # Look for sweet spot in volume discounts
            for i, break_info in enumerate(quantity_breaks):
                if i == len(quantity_breaks) - 1:  # Last break
                    suggested_qty = min(break_info['quantity'], stock_qty)
                    break
                
                # Check if next break offers significant savings (>5%)
                next_break = quantity_breaks[i + 1]
                savings_percent = ((break_info['price_usd'] - next_break['price_usd']) / 
                                 break_info['price_usd'] * 100)
                
                if savings_percent >= 5 and next_break['quantity'] <= stock_qty:
                    continue  # Go to next break
                else:
                    suggested_qty = min(break_info['quantity'], stock_qty)
                    break
        else:
            suggested_qty = moq
        
        return max(suggested_qty, moq)