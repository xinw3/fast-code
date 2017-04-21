package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.IntWritable;

public class SimilarityCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {

	@Override
	protected void reduce(Text key, Iterable<IntWritable> value, Context context)
			throws IOException, InterruptedException {

		int count = 0;
		for (IntWritable val : value) {
			count += val.get();
		}
		context.write(key, new IntWritable(count));
	}
}
