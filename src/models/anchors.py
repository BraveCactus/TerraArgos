from torchvision.models.detection.rpn import AnchorGenerator

from src.config import ANCHOR_RATIOS, ANCHOR_SIZES

def create_anchor_generator(sizes=ANCHOR_SIZES, 
                          ratios=ANCHOR_RATIOS):                
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
