from torchvision.models.detection.rpn import AnchorGenerator

PRESETS = {    
    'small_vehicles': {
        'sizes': ((8,), (16,), (32,), (64,), (128,)),  
        'ratios': ((0.5, 1.0, 2.0),) * 5               
    },    
    
    'medium_vehicles': {
        'sizes': ((16,), (32,), (64,), (128,), (256,)),  
        'ratios': ((0.5, 1.0, 2.0),) * 5                 
    },    
    
    'large_vehicles': {
        'sizes': ((32,), (64,), (128,), (256,), (512,)),  
        'ratios': ((0.5, 1.0, 2.0),) * 5                  
    },    
    
    'default': {
        'sizes': ((32,), (64,), (128,), (256,), (512,)), 
        'ratios': ((0.5, 1.0, 2.0),) * 5                
    }
}

def create_anchor_generator(sizes=((32,), (64,), (128,), (256,), (512,)), 
                          ratios=((0.5, 1.0, 2.0),) * 5):                
    """
    Создает anchor generator для FPN (5 feature maps)
    
    Args:
        sizes: размеры anchor'ов для каждого уровня FPN
        ratios: формы anchor'ов для каждого уровня FPN
    """
    anchor_generator = AnchorGenerator(
        sizes=sizes,
        aspect_ratios=ratios
    )
    
    print("Созданы anchor'ы для FPN (5 уровней):")
    for i, (size, ratio) in enumerate(zip(sizes, ratios)):
        print(f"   Уровень {i+1}: размеры {size}, формы {ratio}")
    
    return anchor_generator

def create_preset_anchor_generator(preset_name='medium_vehicles'):
    """
    Создает anchor generator из готового пресета
    """
    if preset_name not in PRESETS:
        print(f"Пресет {preset_name} не найден, использую 'default'")
        preset_name = 'default'
    
    preset = PRESETS[preset_name]
    
    print(f"Использую пресет: {preset_name}")
    
    return create_anchor_generator(preset['sizes'], preset['ratios'])