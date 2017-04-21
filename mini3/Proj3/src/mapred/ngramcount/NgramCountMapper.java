package mapred.ngramcount;

import java.io.IOException;

import mapred.util.Tokenizer;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobConfigurable;


public class NgramCountMapper extends Mapper<LongWritable, Text, Text, NullWritable> {


	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String[] words = Tokenizer.tokenize(line);
		// for (String word : words)
		// 	context.write(new Text(word), NullWritable.get());
		String keyword;
		Configuration conf = context.getConfiguration();
		String number = conf.get("n");
		int num = Integer.parseInt(number);
		for(int i = 0; i < words.length - num + 1; i++){
			keyword = words[i];
			for(int j = i + 1; j < num + i; j++){
				keyword = keyword + " " + words[j];
			}
			context.write(new Text(keyword), NullWritable.get());
		}

	}
}
