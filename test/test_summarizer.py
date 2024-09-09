# summarizer.py
import extract_sum_mp

paragraphs = [
    "Tea, one of the world's most popular beverages, has a rich history dating back thousands of years. Its origins can be traced to ancient China, where legend has it that Emperor Shen Nong discovered tea in 2737 BCE when leaves from a wild tree blew into his pot of boiling water. From these mythical beginnings, tea cultivation and consumption spread throughout China, becoming an integral part of Chinese culture and medicine. By the Tang Dynasty (618-907 CE), tea had become China's national drink, and intricate ceremonies and customs surrounding its preparation and consumption had developed.",
    "The spread of tea beyond China's borders began in earnest during the 8th century, when Japanese Buddhist monks studying in China brought tea plants and seeds back to Japan. The Japanese embraced tea with enthusiasm, developing their own unique tea culture, including the famous Japanese tea ceremony, known as chanoyu. This highly ritualized practice, influenced by Zen Buddhism, emphasizes harmony, respect, purity, and tranquility. The tea ceremony became not just a way of drinking tea, but a profound cultural and spiritual experience that continues to be practiced and revered in Japan to this day.",
    "Tea reached Europe in the early 17th century, brought by Dutch and Portuguese traders. It quickly gained popularity, particularly in England, where it became a symbol of sophistication and social status. The British East India Company played a crucial role in establishing tea as Britain's national beverage, importing vast quantities from China and later India. Tea's importance in British culture is evident in customs like afternoon tea, a light meal typically eaten between 3 and 5 pm. The tea trade also played a significant role in global economics and politics, notably in events like the Boston Tea Party, which helped spark the American Revolution.",
    "In many parts of the world, tea is more than just a drink; it's a social lubricant and a symbol of hospitality. In countries like Morocco, offering tea to guests is an important cultural tradition. The elaborate preparation and serving of mint tea is a sign of respect and friendship. In Russia, tea is typically served from a samovar, a heated metal container, and drinking tea is a social event that can last for hours. In India, chai wallah's (tea vendors) are a common sight on street corners, serving spiced milk tea to customers throughout the day.",
    "Today, tea continues to evolve and adapt to modern tastes and lifestyles. While traditional teas like black, green, and oolong remain popular, new varieties and blends are constantly being developed. Bubble tea, originating in Taiwan in the 1980s, has become a global phenomenon. Herbal and fruit infusions, often marketed for their health benefits, have gained a significant market share. As our understanding of tea's potential health benefits grows, its popularity shows no signs of waning. From its humble beginnings in ancient China to its current status as a global beverage, tea's journey reflects the interconnectedness of world cultures and the enduring human need for comfort, ritual, and connection."
]

# Example usage of a function from extract_sum_mp
result = extract_sum_mp.mass_extract_summaries(paragraphs)

print('Summarized data:')
for i, res in enumerate(result):
    print(f'{i+1}. {res}')

# calc number of words before and after summarization
print('before summarization:', sum(len(paragraph.split()) for paragraph in paragraphs))

print('after summarization:', sum(len(res.split()) for res in result))