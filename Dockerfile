FROM python:3.6

# @TODO to be replaced with `pip install pommerman`
ADD . /pommerman
RUN cd /pommerman && pip install -e .[extras]
# end @TODO

# Should be replaced with a download inside the container
ADD ./data/tensorboard/20211218T184421-aux-softmax/679 ./pommerman/data/models/deployment

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /pommerman
ENTRYPOINT ["python"]
CMD ["./pommermanLearn/run.py"]
