# MEF-GD
MEF-GD: Multimodal Enhancement and Fusion Network for Garment Designer

**Abstract**: <br>
> In recent years, with advancements in generative models, an increasing number of garment design methods have been proposed. A generative model capable of generating garment images from text and sketches can provide designers with valuable visual references and creative inspiration to aid in the design process. Existing multi-modal garment design methods face the challenge of lacking precise control over the generated results in relation to both sketches and text. In this paper, we propose Multimodal Enhancement and Fusion Network for Garment Design (MEF-GD). Our model inputs image conditions into Stable Diffusion based on ControlNet. On one hand, directly inputting multiple image conditions leads to feature forgetting in deep layers. To resolve this problem, we propose a multi-injection module to more effectively enhance image condition features. On the other hand, ControlNet fuses image conditional features into Stable Diffusion through point-by-point addition, which ignores the interaction between multi-modal features and results in unsatisfactory generated images. To make up this flaw, we introduce content-guided attention for more effective feature fusion and improve the expression of text features. In addition, existing datasets often contain vague textual descriptions of garments. It is difficult to train the model on such a dataset to learn accurate alignment between generated image and the textual descriptions. To tackle this issue, we have designed a text optimization module to improve the quality and clarity of text generation. Compared to existing multi-modal garment design methods, MFE-GD more effectively aligns with the requirements of both text and sketches in generating garment images.





https://github.com/user-attachments/assets/3d2eb3bc-0be6-4176-9c43-83ea840fcb3f




https://github.com/user-attachments/assets/a06ff9f5-4b5e-4b4a-a3eb-82b8ce61b9a2

## TODO
- [ ] training code
- [ ] Pre-trained models
- [ ] Inference code
