"""ASL to ISL Translation Module

Provides cross-lingual translation from American Sign Language (ASL) glosses
to Indian Sign Language (ISL) glosses using rule-based transformations and
quantized LLM backbone.

Usage:
    from src.translation.translator import ASLtoISLTranslator
    
    translator = ASLtoISLTranslator()
    isl_glosses = translator.translate(["HELLO", "MY", "NAME"])
    # Returns: ["NAMASKAR", "MERA", "NAM"]
"""

from src.translation.translator import ASLtoISLTranslator

__all__ = ["ASLtoISLTranslator"]
