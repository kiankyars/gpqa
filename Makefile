# Delegate to paper/
PAPER := paper

build:
	$(MAKE) -C $(PAPER) build

arxiv:
	$(MAKE) -C $(PAPER) arxiv

clean:
	$(MAKE) -C $(PAPER) clean

.PHONY: build arxiv clean
