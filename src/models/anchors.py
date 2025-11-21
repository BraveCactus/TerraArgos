from torchvision.models.detection.rpn import AnchorGenerator

PRESETS = {
    # Для маленьких машин (20-80 пикселей)
    'small_vehicles': {
        'sizes': ((16,), (32,), (64,), (128,)),
        'ratios': ((0.7, 1.0, 1.3),) * 4
    },
    
    # Для средних машин (50-200 пикселей) 
    'medium_vehicles': {
        'sizes': ((32,), (64,), (128,), (256,)),
        'ratios': ((0.7, 1.0, 1.5),) * 4
    },
    
    # Для больших машин (100-400 пикселей)
    'large_vehicles': {
        'sizes': ((64,), (128,), (256,), (512,)),
        'ratios': ((0.8, 1.0, 1.2),) * 4
    },
    
    # Универсальные (подходят для большинства случаев)
    'default': {
        'sizes': ((32,), (64,), (128,), (256,)),
        'ratios': ((0.5, 1.0, 2.0),) * 4
    }
}

def create_anchor_generator(sizes=((32,), (64,), (128,), (256,)), 
                          ratios=((0.5, 1.0, 2.0),) * 4):
    """
    Создает простой anchor generator
    
    Args:
        sizes: размеры anchor'ов в пикселях
        ratios: формы anchor'ов
    """
    anchor_generator = AnchorGenerator(
        sizes=sizes,
        aspect_ratios=ratios
    )
    
    print("Созданы anchor'ы:")
    print(f"   Размеры: {sizes}")
    print(f"   Формы: {ratios}")
    
    return anchor_generator

def create_preset_anchor_generator(preset_name='medium_vehicles'):
    """
    Создает anchor generator из готового пресета
    
    Args:
        preset_name: имя пресета - 'small_vehicles', 'medium_vehicles', 
                    'large_vehicles', 'default'
    """
    if preset_name not in PRESETS:
        print(f"Пресет {preset_name} не найден, использую 'default'")
        preset_name = 'default'
    
    preset = PRESETS[preset_name]
    
    print(f"Использую пресет: {preset_name}")
    
    return create_anchor_generator(preset['sizes'], preset['ratios'])