
def validate_numeric_field(name, value_str, constraints):
    type = constraints['type']
    min = constraints.get('min')
    max = constraints.get('max')
    
    if type == 'float':
        if isinstance(value_str, float):
            value = value_str
        else:
            try:
                value = float(value_str)
            except ValueError:
                message = f"Invalid value for {name}. Expected a float, got {value_str}"
                return False, message
                
    elif type == 'int':
        if isinstance(value_str, int):
            value = value_str
        else:
            try:
                value = int(value_str)
            except ValueError:
                message = f"Invalid value for {name}. Expected an integer, got {value_str}"
                return False, message
        
    if min is not None and value < min:
        message = f"Invalid value for {name}. Expected a value greater than or equal to {min}, got {value}"
        return False, message
    if max is not None and value > max:
        message = f"Invalid value for {name}. Expected a value less than or equal to {max}, got {value}"
        return False, message
    return True, value