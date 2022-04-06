from kora.selenium import wd
wd.get('''https://api-mainnet.magiceden.io/rpc/getListedNFTsByQuery?q={"$match":{"collectionSymbol":"degods"},"$sort":{"takerAmount":1},"$skip":0,"$limit":1}''')