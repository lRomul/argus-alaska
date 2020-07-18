NAME?=argus-alaska

GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

.PHONY: all build stop run attach logs exec run-train

all: stop build run-train logs

build:
	docker build -t $(NAME) .

stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

run:
	docker run --rm -dit \
		$(GPUS_OPTION) \
		--net=host \
		--ipc=host \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		bash
	docker attach $(NAME)

run-train:
	docker run --rm -dit \
		$(GPUS_OPTION) \
		--net=host \
		--ipc=host \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		./distributed_train.sh 8 --experiment oneflip32_b5_001_bitmix_005 --pretrain oneflip32_b5_001

attach:
	docker attach $(NAME)

logs:
	docker logs -f $(NAME)

exec:
	docker exec -it $(NAME) bash
