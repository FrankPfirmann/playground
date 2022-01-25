FROM python:3.6

# @TODO to be replaced with `pip install pommerman`
ADD . /pommerman
RUN cd /pommerman && pip install -e .[extras]
# end @TODO

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /pommerman
ENTRYPOINT ["python"]
CMD ["./pommermanLearn/main.py"]
